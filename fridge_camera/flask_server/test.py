from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return 'Hello'

if __name__ == '__main__':
    app.debug = True
    app.run() # 指定なしだとローカルでいける