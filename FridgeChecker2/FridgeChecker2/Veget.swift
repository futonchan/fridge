//
//  Veget.swift
//  FridgeChecker2
//
//  Created by 矢野大暉 on 2021/05/06.
//

import Foundation
struct Veget {
    private(set) public var name : String
    private(set) public var date : String
    private(set) public var imageName : String
    
    init(name: String, date: String, imageName: String) {
        self.name = name
        self.date = date
        self.imageName = imageName
    }
}
