fn main() {
    println!("Hi, I'm a very dumb program.");
    println!("My entire personality is counting how many times I've said I'm dumb.");

    let mut count = 0;
    while count < 10 {
        count += 1;
        println!("Dumb message #{}", count);
    }

    println!("Okay, I'm done. That was exhausting.");
}
