//
//  PredictionViewController.swift
//  VideoStreamingClient
//
//  Created by Jason Yu on 7/10/24.
//

import UIKit

class PredictionViewController: UIViewController {

    @IBOutlet weak var dimensionLabel: UILabel!

    var dimensions: String? {
        didSet {
            updateUI()
        }
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        updateUI()
    }
    
    private func updateUI() {
        DispatchQueue.main.async {
            self.dimensionLabel.text = self.dimensions
        }
    }
}
