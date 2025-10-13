use minijinja::context;

fn raise_exception(err_text: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(
        minijinja::ErrorKind::SyntaxError,
        err_text,
    ))
}

fn strftime_now(s: String) -> Result<String, minijinja::Error> {
    if s == "%Y-%m-%d" {
        let naive = chrono::Utc::now().naive_utc();
        Ok(format!("{}", naive))
    } else {
        Err(minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            format!("invalid format {}", s),
        ))
    }
}

pub fn chat_template(template: &str, system: &str, prompt: &str) -> Result<String, String> {
    let mut env = minijinja::Environment::new();
    minijinja_contrib::add_to_environment(&mut env);

    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function("raise_exception", raise_exception);
    env.add_function("strftime_now", strftime_now);

    const MAIN: &str = "main";

    //println!("{}", str);
    env.add_template(MAIN, template).unwrap();

    let tmpl = env.get_template(MAIN).unwrap();
    let messages = vec![
        context!(role => "system", content => system),
        context!(role => "user", content => prompt),
    ];
    let tools: Vec<String> = Vec::new();

    tmpl.render(context!(tools => tools, messages => messages))
        .map_err(|e| format!("chat template error {}", e))
}
