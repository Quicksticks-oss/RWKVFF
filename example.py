import RWKVFF

user = "USER"
bot = "RWKV"

context = f'''
The following is a conversation between a sentiant ai named {bot}, and a human user called {user}.
'''

if __name__ == '__main__':
    model = RWKVFF.RWKVFromFile(file='RWKV-5-World-1B5-v2-20231025-ctx4096.pth', tokenizer_json='rwkv_vocab_v20230424', verbose=True, rwkv_version='5') # Updated for v5
    state = model.initialize(context)
    #model.load(filename='save.pt') # Loading model
    while True:
        message, state = model.generate(f"\n\n{user}: {input(' >> ')}\n\n{bot}:", ['\n', '\n\n'], state=state)
        print(message)
        model.save('save.pt', state)