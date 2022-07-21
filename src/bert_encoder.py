import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

"""
Credit to: Alexander Henlein: 
Code taken from: https://github.com/texttechnologylab/SpatialAssociationsInLM
"""

class Encoder:
    def __init__(self):
        pass

    def documents_to_vecs(self, documents: [str]):
        raise NotImplementedError



class BertCLSEncoder(Encoder):
    def __init__(self, version='bert-large-uncased'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(version)
        self.model = AutoModel.from_pretrained(version)
        self.model.eval()

    def documents_to_vecs(self, sentences: [str], mode="CLS", obj=None):
        #print(sentences)
        #print(obj)
        #encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        encoded_input = self.tokenizer(sentences, padding=True, return_tensors="pt")
        #print(encoded_input)
        with torch.no_grad():
            output = self.model(**encoded_input)

        results = []
        for sent_idx, sent in enumerate(output["last_hidden_state"]):  #Kann man garantiert auch eleganter lösen ....
            if mode == "CLS":
                results.append(sent[0])
            elif mode == "AVG":
                results.append(torch.tensor(np.average(sent.numpy(), 0)))
            elif mode == "MAX":
                results.append(torch.tensor(np.max(sent.numpy(), 0)))
            elif mode == "OBJ":
                #print(obj)
                encoded_obj = self.tokenizer.encode(obj, add_special_tokens=False)
                encoded_sent = encoded_input["input_ids"][sent_idx]
                obj_results = []

                for enc_obj in encoded_obj:
                    obj_id = (encoded_sent == enc_obj).nonzero(as_tuple=True)[0][0] #Find for every objid the right index in sent
                    obj_results.append(sent[obj_id].numpy())

                obj_results = np.array(obj_results)
                results.append(torch.tensor(np.average(obj_results, 0)))
        results = torch.stack(results, 0)
        return results

    #For Decoder ...
    def document_to_vec(self, sentences: [str], mode="CLS", obj=None):
        results = []
        for sent in sentences:
            encoded_input = self.tokenizer(sent, truncation=True, return_tensors="pt")
            with torch.no_grad():
                output = self.model(**encoded_input)["last_hidden_state"][0]

            if mode == "CLS":
                results.append(output[0])
            elif mode == "AVG":
                results.append(torch.tensor(np.average(output.numpy(), 0)))
            elif mode == "MAX":
                results.append(torch.tensor(np.max(output.numpy(), 0)))
            elif mode == "OBJ":
                #print(obj)
                encoded_obj = self.tokenizer.encode(obj, add_special_tokens=False)
                encoded_sent = encoded_input["input_ids"][0]
                obj_results = []

                for enc_obj in encoded_obj:
                    obj_id = (encoded_sent == enc_obj).nonzero(as_tuple=True)[0][0] #Find for every objid the right index in sent
                    obj_results.append(output[obj_id].numpy())

                obj_results = np.array(obj_results)
                results.append(torch.tensor(np.average(obj_results, 0)))
        results = torch.stack(results, 0)
        return results

    def sent_encoder(self, sent):
        encoded_input = self.tokenizer(sent, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**encoded_input)
        return output


if __name__ == "__main__":
    enc = BertCLSEncoder()
    res = enc.documents_to_vecs(["Ich bin ein Testsatz."])
    print(res.shape)