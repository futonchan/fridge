import UIKit

struct Veget: Codable {
    var name: String        // 名前
    var count: Int            // 個数
}
//let originalObject = Veget(name: "orange", count: 5)
//
///// ②JSONへ変換
//let encoder = JSONEncoder()
//guard let jsonValue = try? encoder.encode(originalObject) else {
//    fatalError("Failed to encode to JSON.")
//}
//
///// ③JSONデータ確認
//print(String(bytes: jsonValue, encoding: .utf8)!)
//
///// ④JSONから変換
//let decoder = JSONDecoder()
//guard let veget: Veget = try? decoder.decode(Veget.self, from: jsonValue) else {
//    fatalError("Failed to decode from JSON.")
//}
//
///// ⑤最終データ確認
//print("***** 最終データ確認 *****")
//print(veget)
//
