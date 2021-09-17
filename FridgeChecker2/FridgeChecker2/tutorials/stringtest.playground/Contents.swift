import UIKit
import Foundation

// "20210822103022" -> 2021/08/22 10:30:22
func String2Datetime(strDate: String) -> String {
    let date = strDate.prefix(8)
    let y = date.prefix(4)
    let m = date[date.index(date.startIndex, offsetBy: 4)..<date.index(date.startIndex, offsetBy: 6)]
    let d = date[date.index(date.startIndex, offsetBy: 6)..<date.index(date.startIndex, offsetBy: 8)]
    
    let time = String(strDate.suffix(6))
    var time2 = ""
    for i in 0...2 {
        let t = String(date[date.index(time.startIndex, offsetBy: i*2)..<date.index(time.suffix(6).startIndex, offsetBy: i*2+2)])
        if i != 2{
            time2 += t + ":"
        } else {
           time2 += t
        }
    }
    return String(y) + "/" + String(m) + "/" + String(d) + " " + time2
}
print(String2Datetime(strDate: ""))
