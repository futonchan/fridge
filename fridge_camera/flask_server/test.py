from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///testdb.sqlite"
db = SQLAlchemy(app)

class VegetModel(db.Model):
    # テーブル名
    __tablename__ = 'veget_model'

    # カラム情報
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    count = db.Column(db.Integer, nullable=False)

    # def __init__(self, name, count): # なぜinitでこれはいる？
    #     self.name = name
    #     self.count = count

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'count': self.count
        }

    def __repr__(self): # 確認用にprintしてくれる
        return f"Veget(name={name}, count={count})"

@app.route("/", methods=["GET"])
def list_veget():
    vegets = VegetModel.query.all()
    return jsonify({'vegets': [veget.to_dict() for veget in vegets]})

if __name__ == "__main__":
    app.run(debug=True)