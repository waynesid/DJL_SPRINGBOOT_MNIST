package com.AthenaML.DJL001.Controller;

import ai.djl.modality.Classifications;
import com.AthenaML.DJL001.Service.TestingService;
import com.AthenaML.DJL001.Service.TrainingService;
import org.springframework.beans.factory.annotation.Autowired;
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
            return ResponseEntity.ok(ResponseMessage.builder().message("Training Data Done").build());
        }catch (Exception e){
            return ResponseEntity.ok(ResponseMessage.builder().message(e.getMessage()).build());
        }
    }

    @GetMapping("/testing")
    public ResponseEntity<ResponseMessage> predict(){
        try {
            Classifications classifications = testingService.testDataset();
            classifications.setTopK(1);
            return ResponseEntity.ok(ResponseMessage.builder().message(classifications.toString()).build());
        }catch (Exception e){
            return ResponseEntity.ok(ResponseMessage.builder().message(e.getMessage()).build());
        }
    }
}
