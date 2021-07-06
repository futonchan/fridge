//
//  VegetTableViewCell.swift
//  FridgeChecker2
//
//  Created by 矢野大暉 on 2021/05/06.
//

import UIKit

class VegetTableViewCell: UITableViewCell {

    @IBOutlet weak var vegetNameLabel: UILabel!
    @IBOutlet weak var vegetDate: UILabel!
    @IBOutlet weak var vegetImageView: UIImageView!
    @IBOutlet weak var vegetNumLabel: UILabel!
    override func awakeFromNib() {
        super.awakeFromNib()
        // Initialization code
    }

    override func setSelected(_ selected: Bool, animated: Bool) {
        super.setSelected(selected, animated: animated)

        // Configure the view for the selected state
    }

}
