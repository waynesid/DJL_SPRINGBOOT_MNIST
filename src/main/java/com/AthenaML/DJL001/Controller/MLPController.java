package com.AthenaML.DJL001.Controller;

import ai.djl.modality.Classifications;
import com.AthenaML.DJL001.Helper.ResponseMessage;
import com.AthenaML.DJL001.Service.TestingService;
import com.AthenaML.DJL001.Service.TrainingService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * @author Wayne Sidney
 * Created on {18/12/2022}
 */
@RestController
@RequestMapping("/mlp")
public class MLPController {
    @Autowired
    TrainingService trainingService;
    @Autowired
    TestingService testingService;

    @GetMapping("/training")
    public ResponseEntity<ResponseMessage> trainMnist(){
        try {
            trainingService.trainMnistDataSet();
           String message = "***training dataset***";
            return ResponseEntity.status(HttpStatus.OK).body(new ResponseMessage(message));
        }catch (Exception e){
            return ResponseEntity.status(HttpStatus.EXPECTATION_FAILED).body(new ResponseMessage(e.getMessage()));
        }
    }

    @GetMapping("/testing")
    public ResponseEntity<ResponseMessage> predict(){
        try {
            Classifications.Classification classifications = testingService.testDataset();
            classifications.getProbability();
            return ResponseEntity.status(HttpStatus.OK).body(new ResponseMessage(classifications.toString()));
        }catch (Exception e){
            return ResponseEntity.status(HttpStatus.EXPECTATION_FAILED).body(new ResponseMessage(e.getMessage()));
        }
    }
}
