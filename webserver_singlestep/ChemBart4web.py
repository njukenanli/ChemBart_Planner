'''
add samplingh method: top_k, top_p
'''

import os
absdir = os.path.dirname(os.path.abspath(__file__))+"/"
from CBTokenizer import CBTokenizer
from transformers import BartForConditionalGeneration
from transformers import BartConfig
import torch
from torch import nn, optim
import torch.nn.functional as F
import argparse
from copy import deepcopy
from multiprocessing import set_start_method
from typing import *
import random
import math
# Visualization utilities
from utils.show_smiles import reaction_smiles_to_image_base64

# Input standardization
from utils.utils import utils

class ChemBart():
    tokenizer=None
    BartNN=None
    config=None
    def __init__(self, path, dev = "cpu"):
        self.tokenizer=CBTokenizer()
        self.config=BartConfig.from_pretrained(absdir + "config.json")
        self.BartNN=BartForConditionalGeneration(self.config)
        self.dev = torch.device(dev)
        self.BartNN.to(self.dev)
        self.model_path = path
        try:
            self.load_model()
            print("load previous model")
            print(self.model_path)
        except:
            print("new model")
    def trans_to_list(self,l):#<cls> is not included
        out=[]
        for i in range(1,len(l)):
            temp=[0.0]*len(self.tokenizer.vocab)
            temp[l[i]]=1.0
            out.append(temp)
        return out
    def load_model(self):
        #self.BartNN.load_state_dict(torch.load(absdir + 'model/ChemBart.pth',map_location='cpu'))
        self.BartNN.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        #self.BartNN = self.BartNN.cpu()
    def single_train(self,data,epoch=1):#data form: [{"inputs"[]:,"outputs":[]}]
        list_data = [self.trans_to_list(i["outputs"]) for i in data]
        optimizer = torch.optim.AdamW(self.BartNN.parameters(), lr=1e-6, weight_decay=1e-6)
        criterion = torch.nn.BCELoss()
        self.BartNN=self.BartNN.train().cuda()
        for e in range(epoch):
            total_loss_train = 0
            print('epoch: ' ,e,flush=True)
            for i in range(len(data)):
                for j in range(len(data[i]["outputs"])-1):
                    optimizer.zero_grad()
                    outputs=torch.softmax(self.BartNN(input_ids=data[i]['inputs'].reshape(1,len(data[i]['inputs'])).cuda(),return_dict=True,
                                            decoder_input_ids=data[i]["outputs"][0:j+1].reshape(1,j+1).cuda()).logits[0],
                                            dim=1)
                    label = torch.tensor(list_data[i][0:j+1]).cuda()
                    #print(outputs.shape,label.shape)
                    batch_loss=criterion(outputs,label)
                    total_loss_train += batch_loss.item()
                    batch_loss.backward()
                    optimizer.step()
                    #print([x.grad for x in optimizer.param_groups[0]['params']])
                    #print(outputs,label,batch_loss)
            print('Train Loss: {}'.format(round(total_loss_train / len(data) ,3)),flush=True)
            torch.save(self.BartNN.state_dict(), absdir + 'model/ChemBart.pth')
    def paral_train(self,stringlist,epoch=100,batch_size=8,DDP=True):
        self.BartNN=self.BartNN.train()
        #below is distributed dara parallelism...... ......
        if DDP:
            set_start_method("forkserver")
            parser = argparse.ArgumentParser()
            parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
            args = parser.parse_args()
            print(args.local_rank)
            torch.cuda.set_device(args.local_rank)
            torch.cuda.empty_cache()
            torch.distributed.init_process_group(backend="nccl", init_method='env://')
            device = torch.device('cuda', args.local_rank)
            #self.BartNN = self.BartNN.cuda(args.local_rank)
            self.BartNN = self.BartNN.to(device)
            #model paralell
            #self.BartNN= self.BartNN.cuda(args.local_rank*2)
            #for i in range(2,12):
            #    self.BartNN.model.decoder.layers[i] = self.BartNN.model.decoder.layers[i].cuda(args.local_rank*2+1)
            #self.BartNN.encoder = self.BartNN.model.encoder.cuda(args.local_rank*2)
            #self.BartNN = torch.nn.parallel.DistributedDataParallel(self.BartNN,device_ids=[args.local_rank], output_device=args.local_rank)
            self.BartNN = torch.nn.parallel.DistributedDataParallel(self.BartNN,
                    device_ids=[args.local_rank],output_device=args.local_rank)
            print("DDP_init succeeded",flush=True)
            #above is distributed dara parallelism.............
        else:
            device=torch.device("cuda:0")
            self.BartNN = self.BartNN.to(device)
        optimizer = torch.optim.AdamW(self.BartNN.parameters(), lr=1e-6, weight_decay=1e-5)
        criterion = torch.nn.BCELoss()
        datapiece=int(len(stringlist)/500)+1
        for e in range(epoch):
            total_loss_train = 0
            print('epoch: ' ,e,flush=True)
            for count in range(datapiece):
                data=[]
                for s in stringlist[500*count:500*(count+1)]:
                    data.extend(self.tokenizer.gen_train_data_fast(s))
                #print(len(data),type(data[0]))
                #print(len(data))
                print(count,"/",datapiece, len(data), flush = True)
                if DDP:
                    train_sampler = torch.utils.data.distributed.DistributedSampler(data)
                    train_dataloader = torch.utils.data.DataLoader(data,batch_size=batch_size, shuffle=False,sampler=train_sampler)
                else:
                    train_dataloader = torch.utils.data.DataLoader(data,batch_size=batch_size, shuffle=True)
                for train_input, train_label in train_dataloader:
                    #print(train_input, train_label, flush=True)
                    optimizer.zero_grad()
                    train_label = train_label.float().to(device)
                    output = self.BartNN(input_ids=train_input['input_ids'].to(device),
                        attention_mask=train_input["attention_mask"].to(device),
                        decoder_input_ids=train_input['decoder_input_ids'].to(device),
                        decoder_attention_mask=train_input['decoder_attention_mask'].to(device),
                        return_dict=True).logits
                    outputs=torch.softmax(torch.stack([output[i][sum(train_input['decoder_attention_mask'][i])-1] for i in range(len(output))]),dim=1)
                    #print(outputs)
                    if DDP:
                        batch_loss=criterion(outputs,train_label)
                    else:
                        batch_loss=criterion(outputs,train_label)
                    #print(outputs, batch_loss, flush=True)
                    total_loss_train += batch_loss.item()
                    batch_loss.backward()
                    optimizer.step()
                    #print(".",end="",flush=True)
                print('\nPiece train Loss: {}'.format(round(total_loss_train / (count+1) ,3)),flush=True)
                if DDP:
                    if args.local_rank == 0:
                        torch.save(self.BartNN.module.state_dict(), absdir + 'model/ChemBart.pth')
                else:
                    torch.save(self.BartNN.state_dict(), absdir + 'model/ChemBart.pth')
                del data
                del train_dataloader
                del train_sampler
            print('Epoch train Loss: {}'.format(round(total_loss_train/(datapiece+1) ,3)),flush=True)
    '''
    def predict(self, s, decoder_input="<cls>",
                top_k=10, max_len=60, stop_with_sep = True):
        with torch.no_grad():
            inputvector=self.tokenizer.encoder(s)[0].tolist()
            self.BartNN=self.BartNN.eval()
            decodervector = self.tokenizer.encoder(decoder_input)[0][:-1].tolist()
            #outputdict=self.BartNN.generate(inputvector, min_length=0, max_length=60 ,return_dict_in_generate=True ,output_scores=True, 
            #    num_beams=top_k , num_return_sequences=top_k,bos_token_id=1,decoder_start_token_id=1,eos_token_id=3,pad_token_id=0 )
            outputprob=self._beam_search(inputvector,decodervector,
                                        top_k,max_len,stop_with_sep, self.dev)
            #print(outputprob)
            outl=[]
            for i in range(top_k):
                outl.append([self.tokenizer.decoder(outputprob[i][0]),outputprob[i][1]])
            return outl
    '''
    def predict(self, s, decoder_input="<cls>",
                sampling_method='beam', top_k=10, top_p=0.9,
                max_len=60, stop_with_sep=True, num_samples=5, temperature=1.0):
        '''
        for generation.

        Args:
            s: input string
            decoder_input: initial decoder input token (e.g., <cls>)
            sampling_method: 'beam', 'top_k', or 'top_p'
            top_k: number of top tokens to consider in top_k sampling
            top_p: cumulative probability threshold in top_p sampling
            max_len: maximum length of generated sequence
            stop_with_sep: whether to stop when encountering a special token (like <end> or >)
            num_samples: number of sequences to sample
        '''
        with torch.no_grad():
            inputvector = self.tokenizer.encoder(s)[0].tolist()
            self.BartNN = self.BartNN.eval()
            decodervector = self.tokenizer.encoder(decoder_input)[0][:-1].tolist()

            if sampling_method == 'beam':
                outputprob = self._beam_search(inputvector, decodervector, num_samples, max_len, stop_with_sep, self.dev)
            elif sampling_method == 'top_k':
                outputprob = self._top_k_sampling(inputvector, decodervector, top_k, max_len, stop_with_sep, self.dev, num_samples=num_samples, temperature=temperature)
            elif sampling_method == 'top_p':
                outputprob = self._top_p_sampling(inputvector, decodervector, top_p, max_len, stop_with_sep, self.dev, num_samples=num_samples, temperature=temperature)
            else:
                raise ValueError("Invalid sampling method. Choose from 'beam', 'top_k', or 'top_p'.")

            outl = []
            for i in range(min(num_samples, len(outputprob))):
                decoded_text = self.tokenizer.decoder(outputprob[i][0])
                outl.append([decoded_text, outputprob[i][1]])
            return outl
        
    def _beam_search(self,s,decodervector,k,maxlen, stop_with_sep, dev):
        #print(s,decodervector)
        out=torch.softmax(self.BartNN(input_ids=torch.tensor([s]).to(dev),
            decoder_input_ids=torch.tensor([decodervector]).to(dev), return_dict=True).logits,dim=2)
        #print(out)
        templist=[]
        probrec=[]
        for j in range(k):
            m=torch.argmax(out[0][-1]).item()
            ids_no_graph = deepcopy(decodervector)
            ids_no_graph.append(m)
            templist.append([ids_no_graph, float(out[0][-1][m])])
            out[0][-1][m]=0
        decoder_input_ids=[k[0] for k in templist]
        probrec=[k[1] for k in templist]
        #print(decoder_input_ids, probrec)
        #print(decoder_input_ids,probrec)
        endans=[]
        maxlen-=1
        count = 1
        while (maxlen>0 and len(endans)<k):
            #input()
            input_ids = [s for i in range(len(decoder_input_ids))]
            #print(input_ids, decoder_input_ids)
            #print(s,decoder_input_ids)
            out=torch.softmax(self.BartNN(input_ids = torch.tensor(input_ids).to(dev),
                decoder_input_ids = torch.tensor(decoder_input_ids).to(dev),
                return_dict=True).logits,dim=2)
            #print(torch.argmax(out[0][-1]))
            del templist
            templist=[]
            for i in range(len(out)):
                #i traverse through k kind of out logits
                for _ in range(k):
                    #every kind pick k
                    m=torch.argmax(out[i][-1]).item()
                    #print(m)
                    ids_no_graph = deepcopy(decoder_input_ids[i])
                    ids_no_graph.append(m)
                    templist.append([ids_no_graph, float(probrec[i]*out[i][-1][m])])
                    out[i][-1][m]=0
            #print(templist)
            templist.sort(key=lambda x: x[1],reverse=True)
            del decoder_input_ids
            del probrec
            tempcount=k
            decoder_input_ids=[]
            probrec=[]
            count+=1
            i = 0
            for i in range(len(templist)):
                if templist[i][0][-1]==self.tokenizer.vocab["<end>"]:
                    templist[i][1] = templist[i][1]**(1/count)
                    endans.append(templist[i])
                elif stop_with_sep and templist[i][0][-1]==self.tokenizer.vocab[">"]:
                    templist[i][1] = templist[i][1]**(1/count)
                    endans.append(templist[i])
                else :
                    decoder_input_ids.append(templist[i][0])
                    probrec.append(templist[i][1])
                    tempcount-=1
                if tempcount==0 or len(endans)>=k:
                    break
            maxlen-=1
            #print(decoder_input_ids,probrec)
        i = 0
        while (len(endans)<k and i < len(decoder_input_ids)):
            endans.append([decoder_input_ids[i], probrec[i]**(1/count)])
            i += 1
        endans.sort(key=lambda x:x[1],reverse=True)
        return endans
    


    def _top_k_sampling(self, s, decodervector, k, maxlen, stop_with_sep, dev, num_samples=5, temperature=1.0):
        """
        Generate sequences using top-k sampling with temperature
        Returns: list of [sequence, probability] sorted by probability
        """
        results = []
        end_token_id = self.tokenizer.vocab["<end>"]
        sep_token_id = self.tokenizer.vocab[">"]
        
        for _ in range(num_samples):
            current_ids = decodervector[:]  # Start with decoder input
            log_prob = 0.0
            step_count = 0
            
            for step in range(maxlen):
                # Prepare input tensors
                input_tensor = torch.tensor([s]).to(dev)
                decoder_tensor = torch.tensor([current_ids]).to(dev)
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.BartNN(
                        input_ids=input_tensor,
                        decoder_input_ids=decoder_tensor,
                        return_dict=True
                    )
                    logits = outputs.logits[0, -1, :]  # Last token logits
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Get top-k tokens
                topk_probs, topk_indices = torch.topk(F.softmax(logits, dim=-1), k)
                
                # Renormalize top-k probabilities
                topk_probs = topk_probs / topk_probs.sum()
                
                # Sample from top-k
                next_token_idx = torch.multinomial(topk_probs, 1).item()
                next_token = topk_indices[next_token_idx].item()
                token_prob = topk_probs[next_token_idx].item()
                
                # Update log probability
                log_prob += math.log(token_prob)
                step_count += 1
                
                # Add token to sequence
                current_ids.append(next_token)
                
                # Check stopping conditions
                if next_token == end_token_id:
                    break
                if stop_with_sep and next_token == sep_token_id:
                    break
            
            # Calculate normalized probability (geometric mean)
            normalized_prob = math.exp(log_prob / step_count) if step_count > 0 else 0.0
            results.append([current_ids, normalized_prob])
        
        # Sort results by probability
        results.sort(key=lambda x: x[1], reverse=True)
        return results    

    def _top_p_sampling(self, s, decodervector, p, maxlen, stop_with_sep, dev, num_samples=5, temperature=1.0):
        """
        Generate sequences using top-p (nucleus) sampling with temperature
        Returns: list of [sequence, probability] sorted by probability
        """
        results = []
        end_token_id = self.tokenizer.vocab["<end>"]
        sep_token_id = self.tokenizer.vocab[">"]
        
        for _ in range(num_samples):
            current_ids = decodervector[:]  # Start with decoder input
            log_prob = 0.0
            step_count = 0
            
            for step in range(maxlen):
                # Prepare input tensors
                input_tensor = torch.tensor([s]).to(dev)
                decoder_tensor = torch.tensor([current_ids]).to(dev)
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.BartNN(
                        input_ids=input_tensor,
                        decoder_input_ids=decoder_tensor,
                        return_dict=True
                    )
                    logits = outputs.logits[0, -1, :]  # Last token logits
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Convert to probabilities
                probs = F.softmax(logits, dim=-1)
                
                # Sort probabilities in descending order
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Remove tokens with cumulative probability above p
                remove_mask = cumulative_probs > p
                # Always keep at least one token
                remove_mask[1:] = remove_mask[:-1].clone()
                remove_mask[0] = False
                
                # Apply mask to sorted indices
                remove_indices = sorted_indices[remove_mask]
                probs[remove_indices] = 0
                
                # Renormalize probabilities
                if probs.sum() > 0:
                    probs /= probs.sum()
                else:
                    # Fallback: use the top token
                    probs = torch.zeros_like(probs)
                    probs[sorted_indices[0]] = 1.0
                
                # Sample next token
                next_token = torch.multinomial(probs, 1).item()
                token_prob = probs[next_token].item()
                
                # Update log probability
                log_prob += math.log(token_prob)
                step_count += 1
                
                # Add token to sequence
                current_ids.append(next_token)
                
                # Check stopping conditions
                if next_token == end_token_id:
                    break
                if stop_with_sep and next_token == sep_token_id:
                    break
            
            # Calculate normalized probability (geometric mean)
            normalized_prob = math.exp(log_prob / step_count) if step_count > 0 else 0.0
            results.append([current_ids, normalized_prob])
        
        # Sort results by probability
        results.sort(key=lambda x: x[1], reverse=True)
        return results    


class CBRetro():
    def __init__(self, path, dev="cuda:0"):
        self.model = ChemBart(path, dev)
        self.model.BartNN.eval()

    def share_memory(self):
        self.model.BartNN.share_memory()

    def precursor(self, product, lock=None, top_k=10, top_p=0.9, sampling_method='beam', num_samples=10, temperature=1.0):
        if lock is not None:
            lock.acquire()
        product = utils.canonize(product)
        out = self.model.predict("<msk>>>" + product, sampling_method=sampling_method, max_len=512,
                                 decoder_input="<cls>", top_k=top_k, top_p=top_p, num_samples=num_samples, temperature=temperature)
        if lock is not None:
            lock.release()
        for j in range(len(out)):
            ans = out[j][0]
            if ans[-5:] == "<end>":
                out[j][0] = ans[5:-5]
            elif ans[-1] == ">":
                out[j][0] = ans[5:-1]
            else:
                out[j][0] = ans[5:]
        return out

    def reagent(self, reactant, product, n=3, lock=None):
        if lock is not None:
            lock.acquire()
        reactant = utils.canonize(reactant)
        product = utils.canonize(product)
        out = self.model.predict(reactant + "><msk>>" + product, max_len=512 - len(reactant),
                                 decoder_input="<cls>" + reactant + ">", num_samples=n)
        if lock is not None:
            lock.release()
        for j in range(len(out)):
            ans = out[j][0]
            if ans[-1] != ">":
                out[j][0] = ans.split(">")[-1]
            else:
                ans = ans.split(">")[-2]
                if ans[-4:] == "<end":
                    ans = ans[:-4]
                out[j][0] = ans
            if out[j][0][-1] == '.':
                out[j][0] = out[j][0][:-1]
        return out

    def product(self, reactant, reagent, n=3, lock=None):
        if lock is not None:
            lock.acquire()
        reactant = utils.canonize(reactant)
        if reagent is not None:
            reagent = utils.canonize(reagent)
            out = self.model.predict(reactant + ">" + reagent + "><msk>", max_len=512 - len(reactant) - len(reagent),
                                    decoder_input="<cls>" + reactant + ">" + reagent + ">", num_samples=n)
        else:
            out = self.model.predict(reactant + ">><msk>", max_len=512 - len(reactant),
                                    decoder_input="<cls>" + reactant + ">>", num_samples=n)
        if lock is not None:
            lock.release()
        for j in range(len(out)):
            ans = out[j][0]
            if ans[-1] != ">":
                out[j][0] = ans.split(">")[-1]
            else:
                ans = ans.split(">")[-2]
                if ans[-4:] == "<end":
                    ans = ans[:-4]
                out[j][0] = ans
        return out
