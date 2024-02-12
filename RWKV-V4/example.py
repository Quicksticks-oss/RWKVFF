import RWKVFF

user = "USER"
bot = "RWKV"

context = f'''
The following is a conversation between a sentiant ai named {bot}, and a human user called {user}.

'''

if __name__ == '__main__':
    model = RWKVFF.RWKVFromFile(file='RWKV-4-Pile-3B-20221110-ctx4096.pth', tokenizer_json='20B_tokenizer.json', verbose=False)
    state = model.initialize(context)
    model.load(filename='save.pt') # Loading model
    while True:
        message, state = model.generate(f"\n\n{user}: {input(' >> ')}\n\n{bot}:", ['\n'], state=state)
        print(message)
        model.save('save.pt', state)
