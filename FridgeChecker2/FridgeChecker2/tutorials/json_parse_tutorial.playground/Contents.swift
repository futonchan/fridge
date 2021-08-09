import UIKit

// 参考サイト: https://capibara1969.com/2551/
// JSONのパース方法について


struct Vegets: Codable {
    var type: String // "vegets"
    var vegets: [Veget]
}

struct Veget: Codable {
    var name: String        // 名前
    var count: Int            // 個数
}

// json get
// "veget"の中身だけとる(リストになってるjson) <- どうやって？
// 下のやり方でパース

let vegets = [
    Veget(name: "cabbege", count: 5),
    Veget(name: "tomato", count: 6)
]
// jsonぽいオブジェクトをどう作る？ {"vegets": [Veget, Veget, ...]}となっているオブジェクト
//      -> structで定義して値入れてインスタンスにする
let myObject = Vegets(type: "vegets", vegets: vegets)
let encoder = JSONEncoder()
guard let jsonEncodedObject = try? encoder.encode(myObject) else {
    fatalError("Failed json Encode") // printじゃだめ。return, throwする必要あり
    /*
     fatalErrorはコンパイルにどの最適化オプションが指定されていても、通ればプログラムが停止します。
     以降の行が実行されることがない -> guard節でreturnやthrowなどを書かなくて良い
     */
}


// クラスの中でも一要素をEncodeしたいときどうする？
let decoder = JSONDecoder()
guard let jsonDecodedObject = try? decoder.decode(Vegets.self, from: jsonEncodedObject) else { // 「Vegets.self」で構造体のMetatypeにアクセス。Metatypeは(オブジェクト名).Typeで示されるオブジェクトの型情報。
    fatalError("Failed json Decode")
}
print("A")
print(jsonDecodedObject)
