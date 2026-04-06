import os, json
absdir = os.path.dirname(os.path.abspath(__file__))+"/"
import torch
class CBTokenizer():
    def __init__ (self):
        with open(absdir + 'vocab.json') as f:
            self.vocab=json.load(f)
        self.d1 = dict()
        self.d2 = dict()
        self.d5 = dict()
        
        for i in self.vocab:
            if len(i)==2:
                self.d2[i]=self.vocab[i]
            elif len(i)==5: 
                self.d5[i]=self.vocab[i]
            else:
                self.d1[i]=self.vocab[i]
        self.vocabkey=list(self.vocab.keys())
    def encoder (self,s,no0mode=True , alllen = 1024, fail_if_find_unknown = True):
        i=0
        out=[0]*alllen
        mask=[0]*alllen
        outi=0
        hasend=False
        while(True):
            if outi>=alllen:
                print("length > maxlen: ",s)
                return []
            if i>=len(s) :
                if hasend==False:
                    out[outi]=self.vocab["<end>"]
                    mask[outi]=1
                break
            if s[i]==" " :
                i+=1
                continue
            if (i==0) and (len(s)>=5) \
                and (s[i:i+5]=="<cls>"):
                     out[outi]=self.d5[s[i:i+5]]
                     mask[outi]=1
                     outi+=1
                     i+=5
                     continue
            elif i==0: 
                out[outi]=self.d5["<cls>"]
                mask[outi]=1
                outi+=1
            else:
                pass
            if (i<len(s)-4)\
                and (s[i:i+5] in self.d5.keys()):
                out[outi]=self.d5[s[i:i+5]]
                mask[outi]=1
                if s[i:i+5]=="<end>":
                    hasend=True
                    break
                outi+=1
                i+=5
                continue
            if s[i].isdigit() and i>0:
                if s[i-1] == "%" and s[i:i+2].isdigit():
                    out[outi]=self.d2[s[i:i+2]]
                    mask[outi]=1
                    outi+=1
                    i+=2
                    continue
                if (not s[i-1].isdigit())\
                        and s[i-1] != "[" and s[i-1] != ":":
                    out[outi]=self.d1[s[i]]
                    mask[outi]=1
                    outi+=1
                    i+=1
                    continue
                if s[i-1].isdigit():
                    out[outi]=self.d1[s[i]]
                    mask[outi]=1
                    outi+=1
                    i+=1
                    continue
            if (i<len(s)-1) \
                and (s[i:i+2] in self.d2.keys()):
                out[outi]=self.d2[s[i:i+2]]
                mask[outi]=1
                outi+=1
                i+=2
                continue
            if s[i] in self.d1.keys():
                out[outi]=self.d1[s[i]]
                mask[outi]=1
                outi+=1
                i+=1
                continue
            print(s[i:])
            print(s)
            print()
            if fail_if_find_unknown:
                return []
            out[outi]=self.d1["?"]
            mask[outi]=1
            i+=1
            outi+=1
        if no0mode:
            return torch.tensor([out[0:outi+1]])
        else:
            return {"input_ids":torch.tensor(out),"attention_mask":torch.tensor(mask)}
    def gen_train_data_fast(self,s):
        from copy import deepcopy as dcp
        try:
            ans = self.encoder(s)
        except Exception as e:
            print(e)
            return []
        if len(ans) == 0:
            return []
        ans = ans[0].tolist()
        decoder_train = [] 
        temp_id = [0]*1024
        temp_att = [0]*1024
        for i in range(len(ans)-1):
            temp_id[i]=ans[i]
            temp_att[i]=1
            label=[0]*len(self.vocab)
            label[ans[i+1]]=1
            decoder_train.append([torch.tensor(dcp(temp_id)),torch.tensor(dcp(temp_att)),torch.tensor(label)])
        encoder_train=[]
        s1=0
        s2=0
        for i in range(len(ans)):
            if ans[i]==self.vocab[">"] :
                if s1 == 0:
                    s1=i
                else:
                    s2=i
                    break
        temp=[self.vocab["<cls>"],self.vocab["<msk>"],self.vocab[">"]]
        temp.extend(ans[s2:])
        att=[1]*len(temp)+[0]*(1024-len(temp))
        temp.extend([0]*(1024-len(temp)))
        encoder_train.append([torch.tensor(temp),torch.tensor(att)])
        if s2-s1 > 1:
            temp = ans[:s1+1]
            temp.append(self.vocab["<msk>"])
            temp.extend(ans[s2:])
            att=[1]*len(temp)+[0]*(1024-len(temp))
            temp.extend([0]*(1024-len(temp)))
            encoder_train.append([torch.tensor(temp),torch.tensor(att)])
        else:
            encoder_train.append([])
        temp = ans[:s2+1]
        temp.append(self.vocab["<msk>"])
        temp.append(self.vocab["<end>"])
        att=[1]*len(temp)+[0]*(1024-len(temp))
        temp.extend([0]*(1024-len(temp)))
        encoder_train.append([torch.tensor(temp),torch.tensor(att)])
        out=[]
        for i in range(3):
            if i==1 and s2-s1==1:
                continue
            if i==0:
                ran = range(0,s1)
            elif i==1:
                ran = range(s1,s2)
            else:
                ran = range(s2,len(ans)-1)
            for j in ran:
                    out.append([{"input_ids":encoder_train[i][0],
                        "attention_mask":encoder_train[i][1],
                        "decoder_input_ids":decoder_train[j][0],
                        "decoder_attention_mask":decoder_train[j][1]},
                        decoder_train[j][2]])
        return out

    def gen_train_data(self,s):
        from copy import deepcopy as dcp
        try:
            ans = self.encoder(s)
        except Exception as e:
            print(e)
            return []
        if len(ans) == 0:
            return []
        ans = ans[0].tolist()
        decoder_train = []
        temp_id = [0]*1024
        temp_att = [0]*1024
        for i in range(len(ans)-1):
            temp_id[i]=ans[i]
            temp_att[i]=1
            label=[0]*len(self.vocab)
            label[ans[i+1]]=1
            decoder_train.append([{"decoder_input_ids": torch.tensor(dcp(temp_id)),
                "decoder_attention_mask":torch.tensor(dcp(temp_att))},torch.tensor(label)])
        encoder_train=[]
        s1=0
        s2=0
        for i in range(len(ans)):
            if ans[i]==self.vocab[">"] :
                if s1 == 0:
                    s1=i
                else : 
                    s2=i
                    break
        #print(s1,s2)
        encoder_train.append({"input_ids":torch.tensor(ans[0:1]+[self.vocab["<msk>"]]+ans[s1:]+[0]*(1024-len(ans)+s1-2)),
            "attention_mask":torch.tensor([1]*(len(ans)-s1+2)+[0]*(1024-len(ans)+s1-2))})
        #encoder_train.append({"input_ids":torch.tensor(ans[0:s1+1]+[self.vocab["<msk>"]]+[self.vocab[">"]]+[self.vocab["<msk>"]]+[self.vocab["<end>"]]\
        #        +[0]*(1024-s1-5)),"attention_mask":torch.tensor([1]*(s1+5)+[0]*(1024-s1-5))})
        encoder_train.append({"input_ids":torch.tensor(ans[0:1]+[self.vocab["<msk>"]]+[self.vocab[">"]]+[self.vocab["<msk>"]]+ans[s2:]+[0]*(1024-len(ans)+s2-4)),
            "attention_mask":torch.tensor([1]*(len(ans)-s2+4)+[0]*(1024-len(ans)+s2-4))})
        #return (encoder_train[-1]["input_ids"],encoder_train[-1]["attention_mask"])
        if s2-s1 > 1:
            encoder_train.append({"input_ids":torch.tensor(ans[0:s1+1]+[self.vocab["<msk>"]]+ans[s2:]+[0]*(1024-len(ans)+s2-s1-2)),
            "attention_mask":torch.tensor([1]*(len(ans)-(s2-s1)+2)+[0]*(1024-len(ans)+s2-s1-2))})
        else:
            encoder_train.append([])
        encoder_train.append({"input_ids":torch.tensor(ans[0:s2+1]+[self.vocab["<msk>"]]+[ans[-1]]+[0]*(1024-s2-3)),
            "attention_mask":torch.tensor([1]*(s2+3)+[0]*(1024-s2-3))})
        out=[]
        for i in range(4):
            if i==2 and s2-s1 == 1:
                continue
            for j in range(len(ans)-1):
                out.append([{"input_ids":encoder_train[i]["input_ids"],
                    "attention_mask":encoder_train[i]["attention_mask"],
                    "decoder_input_ids":decoder_train[j][0]["decoder_input_ids"],
                    "decoder_attention_mask":decoder_train[j][0]["decoder_attention_mask"]},
                    decoder_train[j][1]])
        return out

    def decoder(self,l,endstop=True,padstop=True):
        s=""
        for i in l:
            s+=self.vocabkey[i]
            if i == self.vocab["<end>"] and endstop:
                break
            if i == self.vocab["<pad>"] and padstop:
                break
        return s

    
