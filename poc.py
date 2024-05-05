from mlx_lm import load, generate
model, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")

source_texts = [
    "Around the world, a great many people cook over open fires and on rudimentary stoves. Wood or (charcoal) are very often used as fuels. When they are burned, clouds of smoke and soot are released. Clean cookstoves, on the other hand, reduce pollution from the combustion of these materials by comparison. So, among other things, they reduce the emissions that are emitted. \n \n Funding this type of project will replace traditional cookstoves that burn charcoal inefficiently and without ventilation, and the gases and smoke emitted can be greatly reduced. In addition, the use of such improved cookstoves brings other positive effects for the sustainable development of the local population in developing and emerging countries (see below).  \n Source: drawdown.org",
    "Less air pollution and thus strain on the lungs; less physical strain from collecting firewood",
]

for source_text in source_texts:
    messages = [ {"role": "system", "content": "You are an English to German translator. Please translate all the following English texts to German"},
                 {"role": "user", "content": source_text}, ]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    prompt = tokenizer.decode(input_ids)
    response = generate(model, tokenizer, prompt=prompt)

    print(source_text)
    print("-------------------------")
    print(response)
    print("=========================")
