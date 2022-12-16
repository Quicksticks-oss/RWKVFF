import RWKVFF

user = "USER"
interface = ":"
bot = "RWKV"
context = f'''
The following is a conversation between a sentiant ai named {bot}, and a human user called {user}.

'''

if __name__ == '__main__':
    model = RWKVFF.RWKVFromFile(file='/media/fusion/Models/rwkv/RWKV-4-Pile-3B-20221110-ctx4096.pth', tokenizer_json='20B_tokenizer.json', verbose=False)
    model.initialize(context)
    model.load(filename='save.pt')
    while True:
        print(model.generate(f"\n\n{user}: {input(' >> ')}\n\n{bot}:", ['\n']))
        model.save(filename='save.pt')
