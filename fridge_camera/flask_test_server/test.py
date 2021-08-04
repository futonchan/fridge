from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
import threading
import time

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///maindb.db"
db = SQLAlchemy(app)

class Vegets(db.Model):
    # テーブル名
    __tablename__ = 'Vegets'

    # カラム情報
    veget = db.Column(db.String(100), nullable=False, primary_key=True)
    count = db.Column(db.Integer, nullable=False)

    def to_dict(self):
        return {
            'veget': self.veget,
            'count': self.count
        }

    def __repr__(self): # 確認用にprintしてくれる
        return f"Veget(veget={veget}, count={count})"

@app.route("/", methods=["GET"])
def list_veget():
    vegets = Vegets.query.all()
    return jsonify({'vegets': [veget.to_dict() for veget in vegets]})

if __name__ == "__main__":
     api_thread = threading.Thread(name='rest_service', target=app.run, args=('0.0.0.0',), kwargs=dict(debug=False))
     api_thread.start()
        
     while True:
        print("Main routine!")
        time.sleep(1)

     api_thread.join()
