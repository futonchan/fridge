import UIKit

class ViewController: UIViewController, UITableViewDelegate, UITableViewDataSource {
    
    var listVegets: [Veget] = []

    @IBOutlet weak var myTableView: UITableView!
    @IBOutlet weak var myButton: UIButton!
    
    // 最初に実行されるやつ
    override func viewDidLoad() {
        super.viewDidLoad()
        myTableView.dataSource = self // TableView使う時の必須1
        myTableView.delegate = self //TableView使う時の必須2
    }
    
    // サーバーURLからJSONとってエンコード
    private func getJSONFromServer(accessUrl: String ,userCompletionHandler: @escaping ([Veget]?, Error?) -> Void) {
        let session = URLSession.shared // セッション情報の取り出し -> Default設定のシングルトンオブジェクトをとってきてる
        
        if let url = URL(string: accessUrl) { // URLじゃない文字列のときnil返す. httpsじゃないときはplistでATSの設定をYESにする。
            let request = URLRequest(url: url)
            
            // こういうクロージャの書き方未だに理解できてない
            // let task = session.dataTask(with: url!) { data, response, error in <- こういうのもわからん
            // URLSessionのdataTaskメソッド。レスポンス受け取り準備
            let task = session.dataTask(with: request, completionHandler: {
                (data:Data?, response: URLResponse?, error: Error?) in
                guard let data = data else {
                    fatalError("Error: None Data") // fatalErrorいつかやめる
                }
                // クラスの中でも一要素をEncodeしたいときどうする？
                let decoder = JSONDecoder()
//                decoder.dateDecodingStrategy =
                guard let jsonDecodedObject = try? decoder.decode([Veget].self, from: data) else { // 「Vegets.self」で構造体のMetatypeにアクセス。Metatypeは(オブジェクト名).Typeで示されるオブジェクトの型情報。
                    fatalError("Failed json Decode") // fatalErrorいつかやめる
                }
                userCompletionHandler(jsonDecodedObject,nil)
            })
            
            task.resume()
        }
    }

    func refreshGetVegetsData() {
        let vegetAllGetURL = "https://6v5cialww3.execute-api.ap-northeast-1.amazonaws.com/beta/vegets"
        getJSONFromServer(accessUrl: vegetAllGetURL, userCompletionHandler: { vegetsJson, error in
          if let vegetsJson = vegetsJson {
            self.listVegets.removeAll()
            for v in vegetsJson {
                self.listVegets.append(v)
            }
          }
        })
    }
    
    @IBAction func pushVeget(_ sender: Any) {
        refreshGetVegetsData()
        myTableView?.reloadData()
    }
    // UITableViewの行の数返す, 必須メソッド
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return listVegets.count
    }
    
    // セルを返す 必須メソッド
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        guard let cell = myTableView.dequeueReusableCell(withIdentifier: "VegetTableViewCell", for: indexPath) as? VegetTableViewCell else {
            fatalError("Dequeue failed: VegetTableViewCell.")
        }
        
        let nameText = self.listVegets[indexPath.row].name
        let numVeget = self.listVegets[indexPath.row].count
        let createdAt = self.listVegets[indexPath.row].created_at // 2021-08-25T04:11:33.22543, String
        cell.vegetNameLabel.text = nameText
        cell.vegetNumLabel.text = String(numVeget)
        cell.vegetImageView.image = UIImage(named: nameText)
        return cell
    }
    
    
    
    // tableViewの項目をタップした時
//    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath){
//        let alertController = UIAlertController(title: "野菜情報編集", message: "" , preferredStyle: UIAlertController.Style.alert)
//    }
}
