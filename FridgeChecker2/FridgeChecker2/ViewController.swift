import UIKit

class ViewController: UIViewController, UITableViewDelegate, UITableViewDataSource {
    
    var vegets: [Veget] = []
    var vegetSameDateNum = [String:Int]()
    var a = 0

    @IBOutlet weak var myTableView: UITableView!
    @IBOutlet weak var myButton: UIButton!
    
    // 最初に実行されるやつ
    override func viewDidLoad() {
        super.viewDidLoad()
        
        myTableView.dataSource = self // TableView使う時の必須1
        myTableView.delegate = self //TableView使う時の必須2
        loadData()
    }
    
    // テーブル？DB？名: vegets
    // vegetsに格納するデータ: Veget(name, date, imageName)
    
    // ここをサーバーからデータとってくるようにするだけ
    // とりあえずDBファイルから取ってこれるように
    func loadData() {
        vegets.append(Veget(name: "キャベツ", date: "2021/6/20", imageName: "cabbage"))
    }
    
    @IBAction func pushVeget(_ sender: Any) {
        loadData()
        print(vegets)
        print(vegetSameDateNum)
        myTableView?.reloadData()
    }
    // UITableViewの行の数返す, 必須メソッド
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return vegets.count
    }
    
    // セルを返す 必須メソッド
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
            
        //let cell = myTableView.dequeueReusableCell(withIdentifier: "MyTableViewCell", for: indexPath)
        
        guard let cell = myTableView.dequeueReusableCell(withIdentifier: "VegetTableViewCell", for: indexPath) as? VegetTableViewCell else {
            fatalError("Dequeue failed: VegetTableViewCell.")
        }
        
        let nameText = vegets[indexPath.row].name
        cell.vegetNameLabel.text = nameText
        
        var dateText = vegets[indexPath.row].date
        if (nameText == "キャベツ") {
            let splitDate = dateText.components(separatedBy: "/")
            print(splitDate[1])
            var month:Int = Int(splitDate[1])!
            month += 1
            dateText = splitDate[0] + "/" + String(month) + "/" + splitDate[2]
        }
        cell.vegetDate.text = dateText
        cell.vegetImageView.image = UIImage(named: vegets[indexPath.row].imageName)
        if(vegetSameDateNum[nameText] != nil){
            vegetSameDateNum[nameText]! += 1
        }
        else{
            vegetSameDateNum[nameText] = 1
        }
        return cell
    }
    
    
    
    // tableViewの項目をタップした時
    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath){
        let alertController = UIAlertController(title: "野菜情報編集", message: "" , preferredStyle: UIAlertController.Style.alert)
    }
}
