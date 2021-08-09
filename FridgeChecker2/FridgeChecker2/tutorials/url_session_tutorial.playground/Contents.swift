import UIKit
import PlaygroundSupport // PlayGroundで非同期処理するためのライブラリ

// URLSesisonについて
// 参考: https://swift.codelly.dev/guide/%E3%83%8D%E3%83%83%E3%83%88%E3%83%AF%E3%83%BC%E3%82%AF/
//      本気で始めるiPhoneアプリ作り

struct Vegets: Codable {
    var vegets: [Veget]
}

struct Veget: Codable {
    var veget: String        // 名前
    var count: Int            // 個数
}

PlaygroundPage.current.needsIndefiniteExecution = true // 非同期処理モード


let session = URLSession.shared // セッション情報の取り出し -> Default設定のシングルトンオブジェクトをとってきてる

if let url = URL(string: "http://hiroki.mydns.jp:5000/") { // URLじゃない文字列のときnil返す. httpsじゃないときはplistでATSの設定をYESにする。
    let request = URLRequest(url: url)
    
    // こういうクロージャの書き方未だに理解できてない
    // let task = session.dataTask(with: url!) { data, response, error in <- こういうのもわからん
    let task = session.dataTask(with: request, completionHandler: { // URLSessionのdataTaskメソッド。レスポンス受け取り準備
        (data:Data?, response: URLResponse?, error: Error?) in
        
        guard let data = data else {
            fatalError("Error: None Data")
        }
        // クラスの中でも一要素をEncodeしたいときどうする？
        let decoder = JSONDecoder()
        guard let jsonDecodedObject = try? decoder.decode(Vegets.self, from: data) else { // 「Vegets.self」で構造体のMetatypeにアクセス。Metatypeは(オブジェクト名).Typeで示されるオブジェクトの型情報。
            fatalError("Failed json Decode")
        }
        print(jsonDecodedObject)
    })
    
    task.resume()
}
