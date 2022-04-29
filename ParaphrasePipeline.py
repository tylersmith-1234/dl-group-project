from transformers import Pipeline

class ParaphrasePipeline(Pipeline):
    def _sanitize_parameters(self, **pipeline_parameters):

        print("pipeline parameters: ")
        print(pipeline_parameters)
        return dict(), dict(), dict()

    def preprocess(self, input_, **preprocess_parameters):
        """Tokenize and convert input into torch encodings"""
        print("preprocess:")

        device = self.device

        if isinstance(input_, str):
            encoding = self.tokenizer(f"paraphrase: {input_}", return_tensors="pt")
        
            # Push tensors to device
            for key, value in encoding.items():
                encoding[key] = value.to(device)
            
            input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
            return {"input_ids": input_ids, "attention_masks": attention_masks}
        
        else:
            raise Exception(f"Unhandled input type: {type(input_)}")

    def _forward(self, input_tensors, **forward_parameters):
        """Generate text"""

        print("_forward:")
        print(input_tensors)
        num_return_sequences = 5
        input_ids, attention_masks = input_tensors["input_ids"], input_tensors["attention_masks"]
        out = self.model.generate(input_ids=input_ids, do_sample=True, attention_mask=attention_masks, max_length=512,
                        top_k=250, top_p=0.99, early_stopping=True, num_return_sequences=num_return_sequences)
        return out

    def postprocess(self, model_outputs, **postprocess_parameters):
        """Decode model generated text"""

        print("postprocess")
        print(model_outputs)
        verbose=True
        out = model_outputs
        predictions = []
        for p in out:
            pred = self.tokenizer.decode(p, skip_special_tokens=True)
            predictions.append(pred)
            if verbose is True:
                print('Prediction: ', pred)
        
        return predictions
        # if verbose is True:
        #     print('Expected: ', row['expected'], "\n")