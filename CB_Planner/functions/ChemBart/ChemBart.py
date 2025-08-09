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
class ChemBart():
    tokenizer=None
    BartNN=None
    config=None
    def __init__(self, dev = "cpu", name = "ChemBart"):
        self.name = name
        self.tokenizer=CBTokenizer()
        self.config=BartConfig.from_pretrained(absdir + "config.json")
        self.BartNN=BartForConditionalGeneration(self.config)
        self.dev = torch.device(dev)
        self.BartNN.to(self.dev)
        try:
            self.load_model()
            print("load previous model")
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
        self.BartNN.load_state_dict(torch.load(absdir + f'model/{self.name}.pth',map_location='cpu'))
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
            self.BartNN = self.BartNN.to(device)
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

    def instruct_sft_parallel(self, stringlist, epoch=100, batch_size=8, DDP=True):
        """
        Parallel instruction SFT:
        - Builds one training item per string using tokenizer.prepare_instruct_sft_data(...)
        - Computes sequence-level CrossEntropy over all decoder time steps at once
        - Masks loss for the "<cls>molecule><task_token>" prefix; learns only on "new_molecule<end>"
        """
        self.BartNN = self.BartNN.train()

        # --- DDP / device setup
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
            self.BartNN = self.BartNN.to(device)
            self.BartNN = torch.nn.parallel.DistributedDataParallel(
                self.BartNN, device_ids=[args.local_rank], output_device=args.local_rank
            )
            is_main = (args.local_rank == 0)
            print("DDP_init succeeded", flush=True)
        else:
            device = torch.device("cuda:0")
            self.BartNN = self.BartNN.to(device)
            is_main = True

        # Use AdamW; higher LR than the token-by-token BCE you had is typical for CE SFT.
        optimizer = torch.optim.AdamW(self.BartNN.parameters(), lr=1e-5, weight_decay=1e-5)

        # Chunk your list like before to limit peak RAM while materializing samples
        datapiece = int(len(stringlist) / 500) + 1

        for e in range(epoch):
            total_loss_train = 0.0
            total_steps = 0
            print('epoch:', e, flush=True)

            for count in range(datapiece):
                # Materialize this slice's dataset
                data = []
                for s in stringlist[500 * count: 500 * (count + 1)]:
                    item = self.tokenizer.prepare_instruct_sft_data(s)
                    if isinstance(item, dict) and item:
                        data.append(item)
                print(count, "/", datapiece, len(data), flush=True)
                if len(data) == 0:
                    continue

                # Sampler / DataLoader
                if DDP:
                    train_sampler = torch.utils.data.distributed.DistributedSampler(data)
                    train_dataloader = torch.utils.data.DataLoader(
                        data, batch_size=batch_size, shuffle=False, sampler=train_sampler
                    )
                else:
                    train_sampler = None
                    train_dataloader = torch.utils.data.DataLoader(
                        data, batch_size=batch_size, shuffle=True
                    )

                # Train loop
                for batch in train_dataloader:
                    optimizer.zero_grad()

                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    decoder_input_ids = batch["decoder_input_ids"].to(device)
                    decoder_attention_mask = batch["decoder_attention_mask"].to(device)
                    labels = batch["labels"].to(device)  # -100 where we ignore loss

                    # HuggingFace BART returns CE loss when labels is provided (with ignore_index=-100)
                    outputs = self.BartNN(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        decoder_attention_mask=decoder_attention_mask,
                        labels=labels,
                        return_dict=True,
                    )
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()

                    total_loss_train += float(loss.item())
                    total_steps += 1

                print('\nPiece train Loss: {}'.format(
                    round(total_loss_train / max(total_steps, 1), 3)
                ), flush=True)

                # Save checkpoint (only once on rank 0 if DDP)
                if is_main:
                    try:
                        state = self.BartNN.module.state_dict() if DDP else self.BartNN.state_dict()
                        torch.save(state, absdir + f'model/ChemBart_sft_{e}.pth')
                    except Exception as se:
                        print("Checkpoint save failed:", se, flush=True)

                # Cleanup
                del data
                del train_dataloader
                if DDP and train_sampler is not None:
                    del train_sampler

            print('Epoch train Loss: {}'.format(
                round(total_loss_train / max(total_steps, 1), 3)
            ), flush=True)


    
    def predict(self, s, decoder_input="<cls>",
                top_k=10, max_len=60, stop_with_sep = True):
        '''
        for generation
        '''
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


class CB_END(nn.Module):
    '''
    this api uses the output of end token
    '''
    def __init__(self, out_type: int,
                 name: str, device: str = "cuda:0",
                 ran: int = 0):
        '''
        out_type:
        1: regression
            ran: if ran<0, range in [-ran,ran]
                    if ran>0, range in [0,ran]
                    if ran = 0, range in R
        2: binary classification
        n>=3: ont-hot-encoding classification with n classes
        '''
        super().__init__()
        self.name = absdir + "model/"+name+'.pth'
        self.tokenizer = CBTokenizer()
        self.type = out_type
        self.config=BartConfig.from_pretrained(absdir + "config.json")
        self.BartNN=BartForConditionalGeneration(self.config)
        self.ran = ran
        if self.type == 1 or self.type == 2:
            self.linear = nn.Linear(1024, 1)
        elif self.type > 2:
            self.linear = nn.Linear(1024, self.type)
        else:
            raise("invalid type!")
        self.device = torch.device(device)
        if os.path.exists(self.name):
            self.load_state_dict(torch.load(self.name,map_location='cpu'))
            print("fine-tuned model")
        elif os.path.exists(absdir + 'model/ChemBart.pth'):
            self.BartNN.load_state_dict(torch.load(absdir + 'model/ChemBart.pth',map_location='cpu'))
            print("pre-trained model")
        else:
            print("new model")
        self.to(self.device)
            
    def forward(self, x):
        last_hidden = self.BartNN(input_ids=x, decoder_input_ids=x, return_dict=True, output_hidden_states=True).decoder_hidden_states[-1][0][-1]
        linear_out = self.linear(F.relu(last_hidden))
        if self.type == 1:
            if self.ran == 0:
                return linear_out[0]
            elif self.ran < 0:
                return torch.tanh(linear_out[0])*(-1)*self.ran
            else:
                return torch.sigmoid(linear_out[0])*self.ran
                #ref: (0,10)
        elif self.type == 2:
            return torch.sigmoid(linear_out[0])
        else:
            return torch.softmax(linear_out, dim = 0)
    def single_train(self, data: list, epoch: int, tr: int, val: int, te: int):
        '''
        data: (one piece of input as smiles string, label)
        label: for regression/ bi-classification, float; for multi-classification, one-hot
        '''
        self.to(self.device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-6, weight_decay=1e-6)
        if self.type == 1:
            criterion = torch.nn.MSELoss()
        else:
            criterion = torch.nn.BCELoss()
        bestval = None
        for i in range(epoch):
            print("epoch", i, flush = True)
            ep_loss = 0.0
            cor = 0.0
            count = 0
            self.train()
            for i in data[0:tr]:
                optimizer.zero_grad()
                inp = self.tokenizer.encoder(i[0])
                if len(inp) == 0:
                    continue
                count += 1
                out = self(inp.to(self.device))
                #print(out,i[1][0],flush = True)
                cor += self._get_acc(out.item() if type(i[1]) == type(1.1) else out.tolist(),i[1])
                label = torch.tensor(i[1]).to(self.device)
                loss = criterion(out,label)
                ep_loss = ep_loss + loss.item()
                loss.backward()
                optimizer.step()
            cor = self._post_proc(cor,count)
            print("epoch loss:{}, train_acc:{},train_count:{}".format(ep_loss,cor,count))
            self.eval()
            corval = 0.0
            count = 0
            with torch.no_grad():
                for i in data[tr:tr+val]:
                    inp = self.tokenizer.encoder(i[0])
                    if len(inp) == 0:
                        continue
                    count += 1
                    out = self(inp.to(self.device))
                    corval += self._get_acc(out.item() if type(i[1]) == type(1.1) else out.tolist(),i[1])
            corval = self._post_proc(corval,count)
            print("validation_acc:",corval,",val_count:",count,flush=True)
            if (bestval is None) or\
                    (self.type == 1 and corval < bestval) or\
                    (self.type > 1 and corval > bestval):
                bestval = corval
                torch.save(self.state_dict(), self.name)
                print("model refreshed!", flush = True)
    def test(self, test_data, return_detail = False):
        acc = 0.0
        self.eval()
        self.to(self.device)
        ans = []
        count = 0
        with torch.no_grad():
            for i in test_data:
                inp = self.tokenizer.encoder(i[0])
                if len(inp) == 0:
                    continue
                count+=1
                out = self(inp.to(self.device))
                acc += self._get_acc(out.item() if type(i[1]) == type(1.1) else out.tolist(),i[1])
                if return_detail:
                    ans.append([i[1],out])
            acc = self._post_proc(acc, count)
            print("test_acc:", acc, flush=True)
        return (acc, ans)
    def _get_acc(self,out,label) -> float:
        if self.type == 2:
            if (out<0.5 and label<0.5) or (out>=0.5 and label>=0.5):
                return 1.0
            else:
                return 0.0
        elif self.type == 1:
            return (out - label)**2
        else:
            return float(torch.argmax(out) == torch.argmax(label))
    def _post_proc(self,acc:float,num:int) -> float:
        acc = acc/num
        if self.type == 1:
            acc = acc**0.5
            #rmse
        return acc
    def ret_x_y_list(self,data):
        assert self.type == 1, "only for regression use"
        ans = []
        with torch.no_grad():
            for i in data:
                out = self(self.tokenizer.encoder(i[0]).to(self.device))
                ans.append([i[1],out])
        return ans

class CB_mul_END(nn.Module):
    '''
    this api uses the output of multi tokens at the end
    '''
    def __init__(self, name: str, device: str = "cuda:0"):
        super().__init__()
        self.name = absdir + "model/"+name+'.pth'
        self.tokenizer = CBTokenizer()
        self.config=BartConfig.from_pretrained(absdir + "config.json")
        self.BartNN=BartForConditionalGeneration(self.config)
        self.linear1 = nn.Linear(1024, 1)
        self.linear2 = nn.Linear(1024, 1)
        self.device = torch.device(device)
        if os.path.exists(self.name):
            self.load_state_dict(torch.load(self.name,map_location='cpu'))
            print("fine-tuned model")
        elif os.path.exists(absdir + 'model/ChemBart.pth'):
            self.BartNN.load_state_dict(torch.load(absdir + 'model/ChemBart.pth',map_location='cpu'))
            print("pre-trained model")
        else:
            print("new model")
            
    def forward(self, x):
        last_hidden1, last_hidden2 = self.BartNN(input_ids=x, decoder_input_ids=x, return_dict=True,
                                                output_hidden_states=True).decoder_hidden_states[-1][0][-2:]
        linear_out1 = self.linear1(F.relu(last_hidden1)) #temperature
        linear_out2 = self.linear2(F.relu(last_hidden2)) #yield
        return torch.cat((linear_out1, linear_out2))
    def single_train(self, data: list, epoch: int, tr: int, val: int, te: int):
        '''
        data: (one piece of input as smiles string, label)
        label: for regression/ bi-classification, float; for multi-classification, one-hot
        '''
        self.to(self.device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-6, weight_decay=1e-6)
        criterion = torch.nn.MSELoss()
        bestval_temp = None
        bestval_yiel = None
        for i in range(epoch):
            print("epoch", i, flush = True)
            ep_loss = 0.0
            cor_temp = 0.0
            cor_yiel = 0.0
            count_temp = 0
            count_yiel = 0
            self.train()
            for i in data[0:tr]:
                optimizer.zero_grad()
                if type(i[1][0]) != type(1.1) and type(i[1][1]) != type(1.1):
                    continue
                inp = self.tokenizer.encoder(i[0])
                if len(inp) == 0:
                    continue
                out = self(inp.to(self.device))
                out_no_grad = out.tolist()
                #print(out,i[1][0],flush = True)
                if type(i[1][0]) == type(1.1) and type(i[1][1]) == type(1.1):
                    cor_temp += (i[1][0] - out_no_grad[0])**2
                    cor_yiel += (i[1][1] - out_no_grad[1])**2
                    count_temp += 1
                    count_yiel += 1
                    label = torch.tensor(i[1]).to(self.device)
                    loss = criterion(out,label)
                elif type(i[1][0]) != type(1.1):
                    cor_yiel += (i[1][1] - out_no_grad[1])**2
                    count_yiel += 1
                    label = torch.tensor(i[1][1]).to(self.device)
                    loss = criterion(out[1], label)
                elif type(i[1][1]) != type(1.1):
                    cor_temp += (i[1][0] - out_no_grad[0])**2
                    count_temp += 1
                    label = torch.tensor(i[1][0]).to(self.device)
                    loss = criterion(out[0], label)
                ep_loss = ep_loss + loss.item()
                loss.backward()
                optimizer.step()
            cor_temp = (cor_temp/count_temp)**0.5
            cor_yiel = (cor_yiel/count_yiel)**0.5
            print("epoch loss:{}\n train_temp_acc:{},train_temp_count:{}"
                  .format(ep_loss,cor_temp,count_temp))
            print("train_yiel_acc:{},train_yiel_count:{}"
                  .format(cor_yiel,count_yiel))
            self.eval()
            cor_temp = 0.0
            cor_yiel = 0.0
            count_temp = 0
            count_yiel = 0
            with torch.no_grad():
                for i in data[tr:tr+val]:
                    if type(i[1][0]) != type(1.1) and type(i[1][1]) != type(1.1):
                        continue
                    inp = self.tokenizer.encoder(i[0])
                    if len(inp) == 0:
                        continue
                    out = self(inp.to(self.device))
                    out_no_grad = out.tolist()
                    if type(i[1][0]) == type(1.1) and type(i[1][1]) == type(1.1):
                        cor_temp += (i[1][0] - out_no_grad[0])**2
                        cor_yiel += (i[1][1] - out_no_grad[1])**2
                        count_temp += 1
                        count_yiel += 1
                    elif type(i[1][0]) != type(1.1):
                        cor_yiel += (i[1][1] - out_no_grad[1])**2
                        count_yiel += 1
                    elif type(i[1][1]) != type(1.1):
                        cor_temp += (i[1][0] - out_no_grad[0])**2
                        count_temp += 1
            cor_temp = (cor_temp/count_temp)**0.5
            cor_yiel = (cor_yiel/count_yiel)**0.5
            print("epoch loss:{}\n val_temp_acc:{},val_temp_count:{}"
                  .format(ep_loss,cor_temp,count_temp))
            print("val_yiel_acc:{},val_yiel_count:{}"
                  .format(cor_yiel,count_yiel))
            if (bestval_temp is None) or\
                (cor_temp < bestval_temp and cor_yiel < bestval_yiel):
                bestval_temp = cor_temp
                bestval_yiel = cor_yiel
                torch.save(self.state_dict(), self.name)
                print("model refreshed!", flush = True)
    def test(self, test_data):
        cor_temp = 0.0
        cor_yiel = 0.0
        count_temp = 0
        count_yiel = 0
        self.eval()
        self.to(self.device)
        ans_temp = []
        ans_yiel = []
        count = 0
        with torch.no_grad():
            for i in test_data:
                if type(i[1][0]) != type(1.1) and type(i[1][1]) != type(1.1):
                    continue
                inp = self.tokenizer.encoder(i[0])
                if len(inp) == 0:
                    continue
                count+=1
                out = self(inp.to(self.device))
                out_no_grad = out.tolist()
                #print(out_no_grad,flush = True)
                if type(i[1][0]) == type(1.1) and type(i[1][1]) == type(1.1):
                    cor_temp += (i[1][0] - out_no_grad[0])**2
                    cor_yiel += (i[1][1] - out_no_grad[1])**2
                    count_temp += 1
                    count_yiel += 1
                    ans_temp.append([i[1][0],out_no_grad[0]])
                    ans_yiel.append([i[1][1],out_no_grad[1]])
                elif type(i[1][0]) != type(1.1):
                    cor_yiel += (i[1][1] - out_no_grad[1])**2
                    count_yiel += 1
                    ans_yiel.append([i[1][1],out_no_grad[1]])
                elif type(i[1][1]) != type(1.1):
                    cor_temp += (i[1][0] - out_no_grad[0])**2
                    count_temp += 1
                    ans_temp.append([i[1][0],out_no_grad[0]])
            cor_temp = (cor_temp/count_temp)**0.5
            cor_yiel = (cor_yiel/count_yiel)**0.5
            print("test_temp_acc:", cor_temp, "test_yiel_acc:", cor_yiel, flush=True)
        return ((cor_temp, cor_yiel), (ans_temp, ans_yiel))

class CB_LSTM(nn.Module):
    '''
    this api uses the output of end token
    '''
    def __init__(self, out_type: int,
                 name: str, device: str = "cuda:0",
                 ran: int = 0):
        '''
        out_type:
        1: regression
            ran: if ran<0, range in [-ran,ran]
                    if ran>0, range in [0,ran]
                    if ran = 0, range in R
        2: binary classification
        n>=3: ont-hot-encoding classification with n classes
        '''
        super().__init__()
        self.name = absdir + "model/"+name+'.pth'
        self.tokenizer = CBTokenizer()
        self.type = out_type
        self.config=BartConfig.from_pretrained(absdir + "config.json")
        self.BartNN=BartForConditionalGeneration(self.config)
        self.ran = ran
        self.lstm = torch.nn.LSTM(1024, 1024, num_layers=1,
                                  bias=True, bidirectional=True)
        if self.type == 1 or self.type == 2:
            self.linear = nn.Linear(2048, 1)
        elif self.type > 2:
            self.linear = nn.Linear(2048, self.type)
        else:
            raise("invalid type!")
        self.device = torch.device(device)
        if os.path.exists(self.name):
            self.load_state_dict(torch.load(self.name,map_location='cpu'))
            print("fine-tuned model")
        elif os.path.exists(absdir + 'model/ChemBart.pth'):
            self.BartNN.load_state_dict(torch.load(absdir + 'model/ChemBart.pth',map_location='cpu'))
            print("pre-trained model")
        else:
            print("new model")
            
    def forward(self, x):
        hidden_seq = self.BartNN(input_ids=x, decoder_input_ids=x, return_dict=True, output_hidden_states=True).decoder_hidden_states[-1][0]
        #print(hidden_seq.shape)
        _, (_, cell) = self.lstm(F.relu(hidden_seq))
        linear_out = self.linear(F.relu(cell.reshape(2048)))
        if self.type == 1:
            if self.ran == 0:
                return linear_out[0]
            elif self.ran < 0:
                return torch.tanh(linear_out[0])*(-1)*self.ran
            else:
                return torch.sigmoid(linear_out[0])*self.ran
                #ref: (0,10)
        elif self.type == 2:
            return torch.sigmoid(linear_out[0])
        else:
             return torch.softmax(linear_out, dim = 0)
    def single_train(self, data: list, epoch: int, tr: int, val: int, te: int):
        '''
        data: (one piece of input as smiles string, label)
        label: for regression/ bi-classification, float; for multi-classification, one-hot
        '''
        self.to(self.device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-6, weight_decay=1e-6)
        if self.type == 1:
            criterion = torch.nn.MSELoss()
        else:
            criterion = torch.nn.BCELoss()
        bestval = None
        for i in range(epoch):
            print("epoch", i, flush = True)
            ep_loss = 0.0
            cor = 0.0
            count = 0
            self.train()
            for i in data[0:tr]:
                optimizer.zero_grad()
                inp = self.tokenizer.encoder(i[0])
                if len(inp) == 0:
                    continue
                count += 1
                out = self(inp.to(self.device))
                #print(out,i[1][0],flush = True)
                cor += self._get_acc(out.item() if type(i[1]) == type(1.1) else out.tolist(),i[1])
                label = torch.tensor(i[1]).to(self.device)
                loss = criterion(out,label)
                ep_loss = ep_loss + loss.item()
                loss.backward()
                optimizer.step()
            cor = self._post_proc(cor,count)
            print("epoch loss:{}, train_acc:{},train_count:{}".format(ep_loss,cor,count))
            self.eval()
            corval = 0.0
            count = 0
            with torch.no_grad():
                for i in data[tr:tr+val]:
                    inp = self.tokenizer.encoder(i[0])
                    if len(inp) == 0:
                        continue
                    count += 1
                    out = self(inp.to(self.device))
                    corval += self._get_acc(out.item() if type(i[1]) == type(1.1) else out.tolist(),i[1])
            corval = self._post_proc(corval,count)
            print("validation_acc:",corval,",val_count:",count,flush=True)
            if (bestval is None) or\
                    (self.type == 1 and corval < bestval) or\
                    (self.type > 1 and corval > bestval):
                bestval = corval
                torch.save(self.state_dict(), self.name)
                print("model refreshed!", flush = True)
    def test(self, test_data, return_detail = False):
        acc = 0.0
        self.eval()
        self.to(self.device)
        ans = []
        count = 0
        with torch.no_grad():
            for i in test_data:
                inp = self.tokenizer.encoder(i[0])
                if len(inp) == 0:
                    continue
                count+=1
                out = self(inp.to(self.device))
                acc += self._get_acc(out.item() if type(i[1]) == type(1.1) else out.tolist(),i[1])
                if return_detail:
                    ans.append([i[1],out])
            acc = self._post_proc(acc, count)
            print("test_acc:", acc, flush=True)
        return (acc, ans)
    def _get_acc(self,out,label) -> float:
        if self.type == 2:
            if (out<0.5 and label<0.5) or (out>=0.5 and label>=0.5):
                return 1.0
            else:
                return 0.0
        elif self.type == 1:
            return (out - label)**2
        else:
            return float(torch.argmax(out) == torch.argmax(label))
    def _post_proc(self,acc:float,num:int) -> float:
        acc = acc/num
        if self.type == 1:
            acc = acc**0.5
            #rmse
        return acc
    def ret_x_y_list(self,data):
        assert self.type == 1, "only for regression use"
        ans = []
        with torch.no_grad():
            for i in data:
                out = self(self.tokenizer.encoder(i[0]).to(self.device))
                ans.append([i[1],out])

class CB_MCTS():
    def __init__(self, dev = "cpu"):
        self.core = CB_END(out_type = 1, name = "CB_MCTS" , device = dev, ran = 0)
      
    def policy(self, input_list):
        outlist = torch.stack([self.core(i.to(self.core.device)) for i in input_list])
        p = torch.softmax(outlist, dim = 0)
        return p

    def value(self, smi):
        v = torch.tanh(self.core(smi.to(self.core.device)))
        return v
      
    def forward(self, smi, input_list):
        v = torch.tanh(self.core(smi.to(self.core.device)))
        outlist = torch.stack([self.core(i.to(self.core.device)) for i in input_list])
        p = torch.softmax(outlist, dim = 0)
        return v,p

    def single_train(self, data: list, epoch: int, tr: int, val: int, te: int):
        self.core.to(self.core.device)
        optimizer = torch.optim.AdamW(self.core.parameters(), lr=1e-6, weight_decay=1e-6)
        criterion = torch.nn.MSELoss()
        bestval = None
        for i in range(epoch):
            print("epoch", i, flush = True)
            loss_v = 0.0
            loss_p = 0.0
            count = 0
            self.core.train()
            for i in data[0:tr]:
                optimizer.zero_grad()
                smi = self.tokenizer.encoder(i[0][0])
                inputlist = [self.tokenizer.encoder(k) for k in i[0][1]]
                if len(smi) == 0:
                    continue
                status = 1
                for k in inputlist:
                    if len(k) == 0:
                        status = 0
                        break
                if status == 0:
                    continue
                count += 1
                v, p = self(smi = smi, input_list = inputlist)
                #print(out,i[1][0],flush = True)
                label_v = torch.tensor(i[1][0]).to(self.device)
                label_p = torch.tensor(i[1][1]).to(self.device)
                lv = criterion(v,label_v)
                lp = criterion(p,label_p)
                loss_v += lv.item()
                loss_p += lp.item()
                loss = lv + lp
                loss.backward()
                optimizer.step()
            print("loss v:{}, loss p:{},train_count:{}".format(loss_v,loss_p,count))
            self.core.eval()
            loss_v = 0.0
            loss_p = 0.0
            count = 0
            with torch.no_grad():
                for i in data[tr:tr+val]:
                    smi = self.tokenizer.encoder(i[0][0])
                    inputlist = [self.tokenizer.encoder(k) for k in i[0][1]]
                    if len(smi) == 0:
                        continue
                    status = 1
                    for k in inputlist:
                        if len(k) == 0:
                            status = 0
                            break
                    if status == 0:
                        continue
                    count += 1
                    v, p = self(smi = smi, input_list = inputlist)
                    label_v = torch.tensor(i[1][0]).to(self.device)
                    label_p = torch.tensor(i[1][1]).to(self.device)
                    lv = criterion(v,label_v)
                    lp = criterion(p,label_p)
                    loss_v += lv.item()
                    loss_p += lp.item()
            print("val loss v", loss_v, "val loss p", loss_p,",val_count:",count,flush=True)
            val_loss = loss_v + loss_p
            if (bestval is None) or val_loss<bestval:
                bestval = val_loss
                torch.save(self.core.state_dict(), self.core.name)
                print("model refreshed!", flush = True)
    def test(self, test_data):
            self.core.to(self.device)
            self.core.eval()
            loss_v = 0.0
            loss_p = 0.0
            count = 0
            with torch.no_grad():
                for i in test_data:
                    smi = self.tokenizer.encoder(i[0][0])
                    inputlist = [self.tokenizer.encoder(k) for k in i[0][1]]
                    if len(smi) == 0:
                        continue
                    status = 1
                    for k in inputlist:
                        if len(k) == 0:
                            status = 0
                            break
                    if status == 0:
                        continue
                    count += 1
                    v, p = self(smi = inp, input_list = inputlist)
                    label_v = torch.tensor(i[1][0]).to(self.device)
                    label_p = torch.tensor(i[1][1]).to(self.device)
                    lv = criterion(v,label_v)
                    lp = criterion(p,label_p)
                    loss_v += lv.item()
                    loss_p += lp.item()
            print("loss v", loss_v, "loss p", loss_p,",count:",count,flush=True)

class CB_Regression(nn.Module):
    class RegData():
        def __init__(self, data, tokenizer, maxlen = 1024):
            self.data = data
            self.tokenizer = tokenizer
            self.maxlen = maxlen
        def __getitem__(self, index):
            return self.tokenizer.encoder(self.data[index][0], alllen = self.maxlen, no0mode = False), self.data[index][1]
        def __len__(self):
            return len(self.data)
        def shuffle(self):
            random.shuffle(self.data)
    class DataLoader():
        def __init__(self, data, batch_size, shuffle = True):
            if shuffle:
                data.shuffle()
            self.data = data
            self.batch_size = batch_size
            self.id = 0
            self.len = len(data)
        def __iter__(self):
            return self
        def __next__(self):
            if self.id < self.len:
                count = 0
                x = []
                msk = []
                lab = []
                while (count < self.batch_size and self.id < self.len):
                    inp, label = self.data[self.id]
                    if len(inp) == 0:
                        continue
                    x.append(inp["input_ids"])
                    msk.append(inp["attention_mask"])
                    lab.append(label)
                    self.id += 1
                    count += 1
                return ((torch.stack(x),torch.stack(msk)),torch.tensor(lab))
            else:
                raise StopIteration
    def __init__(self, name: str, label_num: int, device: str):
        super().__init__()
        self.label_num = label_num
        self.name = absdir + "model/"+name+'.pth'
        self.tokenizer = CBTokenizer()
        self.config=BartConfig.from_pretrained(absdir + "config.json")
        self.BartNN=BartForConditionalGeneration(self.config)
        self.linear_heads = nn.ModuleList([nn.Linear(1024, 1) for i in range(label_num)])
        self.device = torch.device(device)
        if os.path.exists(self.name):
            self.load_state_dict(torch.load(self.name,map_location='cpu'))
            print("fine-tuned model")
        elif os.path.exists(absdir + 'model/ChemBart.pth'):
            self.BartNN.load_state_dict(torch.load(absdir + 'model/ChemBart.pth',map_location='cpu'))
            print("pre-trained model")
        else:
            print("new model")
            
    def forward(self, x, attention_mask = None):
        hidden_list = torch.stack([i[-self.label_num:] for i in \
                self.BartNN(input_ids=x, decoder_input_ids=x,
                attention_mask = attention_mask, decoder_attention_mask = attention_mask,
                return_dict=True, output_hidden_states=True).decoder_hidden_states[-1]\
                ])
        linear_out = torch.stack([torch.cat([self.linear_heads[j](hidden_list[i][j])\
                        for j in range(self.label_num)])\
                        for i in range(len(hidden_list))])
        return linear_out

    def fit(self, data: list, epoch: int, batch_size:int, tr: int, val: int, te: int, id_maxlen: int = 1024):
        self.to(self.device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-6, weight_decay=1e-6)
        criterion = torch.nn.MSELoss()
        bestloss = None
        tr_dataset = self.RegData(data[0:tr], self.tokenizer, maxlen = id_maxlen)
        val_dataset = self.RegData(data[tr:tr+val], self.tokenizer, maxlen = id_maxlen)
        for e in range(epoch):
            dataloader = self.DataLoader(tr_dataset, batch_size = batch_size)
            print("epoch", e, flush = True)
            ep_loss = 0.0
            reslist = [[] for k in range(self.label_num)]
            self.train()
            print("train")
            for inp, label in dataloader:
                optimizer.zero_grad()
                out = self(inp[0].to(self.device),attention_mask = inp[1].to(self.device))
                out_no_grad = out.tolist()
                for outcome in range(len(out_no_grad)):
                    for item in range(self.label_num):
                        if type(label[outcome][item]) != type(torch.tensor(1.1)):
                            label[outcome][item] = 0.0
                            out[outcome][item] = 0.0
                            out_no_grad[outcome][item] = 0.0
                            continue
                        reslist[item].append((out_no_grad[outcome][item],label[outcome][item].item()))
                loss = criterion(out, label.to(self.device))
                ep_loss += loss.item()
                loss.backward()
                optimizer.step()
            for idx in range(len(reslist)):
                print("regression task", idx, ": rmse =", self.RMSE(reslist[idx]))
            print("train loss:", ep_loss, flush = True)
            self.eval()
            print("validation")
            reslist = [[] for k in range(self.label_num)]
            dataloader = self.DataLoader(val_dataset, batch_size = batch_size)
            with torch.no_grad():
                for inp, label in dataloader:
                    out = self(inp[0].to(self.device),attention_mask = inp[1].to(self.device))
                    out_no_grad = out.tolist()
                    for outcome in range(len(out_no_grad)):
                        for item in range(self.label_num):
                            if type(label[outcome][item]) != type(torch.tensor(1.1)):
                                continue
                            reslist[item].append((out_no_grad[outcome][item],label[outcome][item].item()))
            RMSE_sum = 0.0
            for idx in range(len(reslist)):
                item_RMSE = self.RMSE(reslist[idx])
                RMSE_sum += item_RMSE
                print("regression task", idx, ": rmse =", item_RMSE)
            if (bestloss is None) or\
                (RMSE_sum < bestloss):
                bestloss = RMSE_sum
                torch.save(self.state_dict(), self.name)
                print("model refreshed!", flush = True)
    def RMSE(self, l):
        s = 0.0
        for x,y in l:
            s += (x-y)**2
        s /= len(l)
        s = s**0.5
        return s
    def test(self, test_data, batch_size, id_maxlen = 1024):
        self.eval()
        self.to(self.device)
        dataset = self.RegData(test_data, self.tokenizer, maxlen = id_maxlen)
        reslist = [[] for k in range(self.label_num)]
        dataloader = self.DataLoader(dataset, batch_size = batch_size)
        with torch.no_grad():
            for inp, label in dataloader:
                out = self(inp[0].to(self.device),attention_mask = inp[1].to(self.device))
                out_no_grad = out.tolist()
                for outcome in range(len(out_no_grad)):
                    for item in range(self.label_num):
                        if type(label[outcome][item]) != type(torch.tensor(1.1)):
                            continue
                        reslist[item].append((out_no_grad[outcome][item],label[outcome][item].item()))
        RMSE_list = []
        for idx in range(len(reslist)):
            item_RMSE = self.RMSE(reslist[idx])
            RMSE_list.append(item_RMSE)
        return (RMSE_list, reslist)

class CB_multi_task_sep_regression(nn.Module):
    '''
    Your input should be: ["<cls>molecule_smiles<task_token><end>", label]
    where <task_token> is <n00> <n01> <n02> <n03> <n04> representing different properties, 
    and label is float number
    '''
    def __init__(self, name: str, num_feature: int, binary_classification: bool, device: str = "cuda:0"):
        '''
        out_type:
        1: regression
        2: binary classification
        '''
        super().__init__()
        self.name = absdir + "model/"+name+'.pth'
        self.tokenizer = CBTokenizer()
        self.config=BartConfig.from_pretrained(absdir + "config.json")
        self.BartNN=BartForConditionalGeneration(self.config)
        self.linear_list = nn.ModuleList([nn.Linear(1024, 1) for _ in range(num_feature)])
        self.classification = binary_classification
        self.device = torch.device(device)
        if os.path.exists(self.name):
            self.load_state_dict(torch.load(self.name,map_location='cpu'))
            print("fine-tuned model")
        elif os.path.exists(absdir + 'model/ChemBart.pth'):
            self.BartNN.load_state_dict(torch.load(absdir + 'model/ChemBart.pth',map_location='cpu'))
            print("pre-trained model")
        else:
            print("new model")
        self.to(self.device)
            
    def forward(self, x, category: int):
        last_hidden = (self.BartNN(input_ids=x, 
                                  decoder_input_ids=x, 
                                  return_dict=True, 
                                  output_hidden_states=True)
                                  .decoder_hidden_states[-1][0][-2]
                                  # get task token output
                        )
        linear_out = self.linear_list[category](F.leaky_relu(last_hidden))
        if self.classification:
            linear_out = torch.sigmoid(linear_out)
        return linear_out[0]

    def _get_category(self, smiles):
        if "<n00>" in smiles:
            category = 0
        elif "<n01>" in smiles:
            category = 1
        elif "<n02>" in smiles:
            category = 2
        elif "<n03>" in smiles:
            category = 3
        elif "<n04>" in smiles:
            category = 4
        else:
            raise ValueError( '''
                    Your input should be: ["<cls>molecule_smiles<task_token><end>", label]
                    where <task_token> is <n00> <n01> <n02> <n03> <n04> representing different properties, 
                    and label is float number
                    ''')
        return category


    def single_train(self, data: list, epoch: int, tr: int, val: int, grad_accumulate: int = 4):
        '''
        data: (one piece of input as smiles string, label)
        label: for regression/ bi-classification, float
        '''
        self.to(self.device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-6, weight_decay=1e-6)
        if self.classification:
            criterion = torch.nn.BCELoss()
        else:
            criterion = torch.nn.MSELoss()
        bestval = None
        for i in range(epoch):
            print("epoch", i, flush = True)
            ep_loss = 0.0
            cor = 0.0
            count = 0
            self.train()
            for i in data[0:tr]:
                category = self._get_category(i[0])
                inp = self.tokenizer.encoder(i[0])
                if len(inp) == 0:
                    continue
                count += 1
                out = self(inp.to(self.device), category)
                #print(out,i[1][0],flush = True)
                cor += self._get_acc(out.item() if type(i[1]) == type(1.1) else out.tolist(),i[1])
                label = torch.tensor(i[1]).to(self.device)
                loss = criterion(out,label) / grad_accumulate
                ep_loss = ep_loss + loss.item()
                loss.backward()
                if count % grad_accumulate == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            if count % grad_accumulate != 0:
                optimizer.step()
                optimizer.zero_grad()
            cor = self._post_proc(cor,count)
            print("epoch loss:{}, train_acc:{},train_count:{}".format(ep_loss,cor,count))
            self.eval()
            corval = 0.0
            count = 0
            with torch.no_grad():
                for i in data[tr:tr+val]:
                    category = self._get_category(i[0])
                    inp = self.tokenizer.encoder(i[0])
                    if len(inp) == 0:
                        continue
                    count += 1
                    out = self(inp.to(self.device), category)
                    corval += self._get_acc(out.item() if type(i[1]) == type(1.1) else out.tolist(),i[1])
            corval = self._post_proc(corval,count)
            print("validation_acc:",corval,",val_count:",count,flush=True)
            if (bestval is None) or\
                    (self.type == 1 and corval < bestval) or\
                    (self.type > 1 and corval > bestval):
                bestval = corval
                torch.save(self.state_dict(), self.name)
                print("model refreshed!", flush = True)
    
    def pred_one_instance(self, mol: str) -> float:
        category = self._get_category(mol)
        inp = self.tokenizer.encoder(mol)
        if len(inp) == 0:
            print("Warning! 0 token is available.")
            return 0.0
        with torch.no_grad():
            out = self(inp.to(self.device), category)
        return float(out)
    
    def test(self, test_data, return_detail = False):
        acc = 0.0
        self.eval()
        self.to(self.device)
        ans = []
        count = 0
        with torch.no_grad():
            for i in test_data:
                category = self._get_category(i[0])
                inp = self.tokenizer.encoder(i[0])
                if len(inp) == 0:
                    continue
                count+=1
                out = self(inp.to(self.device), category)
                acc += self._get_acc(out.item() if type(i[1]) == type(1.1) else out.tolist(),i[1])
                if return_detail:
                    ans.append([i[1],out])
            acc = self._post_proc(acc, count)
            print("test_acc:", acc, flush=True)
        return (acc, ans)
    def _get_acc(self,out,label) -> float:
        if self.type == 2:
            if (out<0.5 and label<0.5) or (out>=0.5 and label>=0.5):
                return 1.0
            else:
                return 0.0
        elif self.type == 1:
            return (out - label)**2
        else:
            return float(torch.argmax(out) == torch.argmax(label))
    def _post_proc(self,acc:float,num:int) -> float:
        acc = acc/num
        if self.type == 1:
            acc = acc**0.5
            #rmse
        return acc
    def ret_x_y_list(self,data):
        assert self.type == 1, "only for regression use"
        ans = []
        with torch.no_grad():
            for i in data:
                out = self(self.tokenizer.encoder(i[0]).to(self.device))
                ans.append([i[1],out])
        return ans