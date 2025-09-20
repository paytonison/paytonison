/**
 * FrontendPortal
 * --------------
 * Lightweight helper to:
 * 1. Persist per-project configuration
 * 2. Launch VS Code for the selected project root
 * 3. Offer a simple observer mechanism for lifecycle hooks
 *
 * This aims to stay minimal and production-ready instead of a demo script.
 */

import * as path from "path";
import * as fs from "fs-extra";
import { spawn } from "child_process";

interface FrontendConfig {
  telemetry: boolean;
  ui: { theme: string };
}

const defaultConfig: FrontendConfig = {
  telemetry: true,
  ui: { theme: "default" }
};

const CONFIG: FrontendConfig = {
  ...defaultConfig,
  ...JSON.parse(process.env.HYPERNEXTGEN_CONFIG || "{}")
};

export class FrontendPortal {
  private static _instance: FrontendPortal | null = null;
  private _observers: Array<() => void> = [];

  constructor(private projectRoot: string = process.cwd()) {
    if (FrontendPortal._instance) {
      return FrontendPortal._instance;
    }
    FrontendPortal._instance = this;

    console.log("[frontend-portal] Initialising with config:", CONFIG);
    this.ensureLicense();
  }

  /**
   * Launch VS Code (macOS) for the current project root.
   * Uses `open -a "Visual Studio Code"` which is the most
   * reliable way on macOS without depending on PATH setup.
   */
  launch(): void {
    const child = spawn("open", ["-a", "Visual Studio Code", this.projectRoot], {
      stdio: "inherit"
    });

    child.on("exit", (code) => {
      console.log(`[frontend-portal] VS Code exited with code ${code}`);
      this._observers.forEach((fn) => fn());
    });
  }

  addObserver(fn: () => void): void {
    this._observers.push(fn);
  }

  /**
   * Write a basic LICENSE file if one does not exist.
   * This is mainly for new projects bootstrapped via scripts.
   */
  private ensureLicense(): void {
    const licensePath = path.join(this.projectRoot, "LICENSE.txt");
    if (!fs.existsSync(licensePath)) {
      fs.writeFileSync(
        licensePath,
        [
          "HyperNextGen License",
          "--------------------",
          "This project is licensed under the HyperNextGen license.",
          "For full terms see https://example.com/license"
        ].join("\n")
      );
      console.log("[frontend-portal] LICENSE.txt created.");
    }
  }
}

/* ---------------------------------------------------------------------- */
/* CLI Entrypoint                                                         */
/* ---------------------------------------------------------------------- */
if (require.main === module) {
  const portal = new FrontendPortal();
  portal.launch();
}
