import os
import numpy as np
import argparse
import glob
def move_metric_to_main_node(log_path,log_name,main_server_path='metric'):
    os.makedirs(main_server_path,exist_ok=True)
    split_name,ext=log_name[:-4],log_name[-4:]
    pre_len=len(glob.glob(os.path.join(main_server_path,split_name+"*")))
    cur_index=pre_len+1
    os.system('cp -rf {} {}'.format(os.path.join(log_path,log_name),os.path.join(main_server_path,split_name+str(cur_index)+ext)))


def get_offline_protocol_index(class_=10):
    eval_list={'offline':[],'online':[],'accuracy':[],'backward':[],'forward':[]}
    count=0
    for i in range(class_):
        for k in range(class_):
            if(i==k):
                eval_list['offline'].append(count)
                eval_list['accuracy'].append(count)
            if(i+1==k):
                eval_list['online'].append(count)
            if(i>k):
                eval_list['backward'].append(count)
                eval_list['accuracy'].append(count)
            if(i<k):
                eval_list['forward'].append(count)
                
            count=count+1
    assert len(eval_list['offline'])==class_
    assert len(eval_list['online'])==class_-1
    assert len(eval_list['backward'])==int(class_*(class_-1)/2)
    assert len(eval_list['forward'])==int(class_*(class_-1)/2)
    assert len(eval_list['accuracy'])==int(class_*(class_+1)/2)
    return eval_list

def get_online_protocol_index(class_=10):
    eval_list={'online':[],'forward':[]}
    # to match our script, since 0-10 is 0
    count=class_
    for i in range(class_):
        for k in range(class_):
            if(i+1==k):
                eval_list['online'].append(count)
            if(i<k):
                eval_list['forward'].append(count)
                
            count=count+1
    assert len(eval_list['online'])==class_-1
    assert len(eval_list['forward'])==int(class_*(class_-1)/2)
    return eval_list
if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--split")
    argparser.add_argument("--timestamp",type=int,default=10)
    argparser.add_argument("--verbose",type=int,default=0)
    argparser.add_argument("--move",type=int,default=0)
    argparser.add_argument("--train_eval",type=int,default=0) # whether the code also include the evaluation of training set as well 

    args = argparser.parse_args()

    logpath='../{}/log/'.format(args.split)
    log_list=sorted(os.listdir(logpath))

    for name in log_list:
        if(name.split('.')[-1]!='txt'):
            continue
        result_list=[]
        log_file_name=os.path.join(logpath,name)
        file=open(log_file_name, 'r')
        while(True):
            try:
                line=file.readline()
            except:
                break
            if('Top1_Acc_Stream/eval_phase/test_stream/Task0' in line):
                result_list.append(float(line.split()[-1]))
            if not line:
                break
        file.close()
        if(args.move==1):
            move_metric_to_main_node(logpath,name)
        if(args.train_eval==1):
            try:
                lowerIndex=[i for i in range(10,200,20)]
                upperIndex=[i for i in range(21,210,20)]
                eval_index=[k for i in range(len(lowerIndex)) for k in range(lowerIndex[i],upperIndex[i]-1) ]
                result_list=np.array(result_list)[eval_index]
            except:
                print('#################################################')
                print('Skipping {} for removing train result'.format(name))
                print('#################################################')
        if(len(result_list)!=int(args.timestamp*args.timestamp)):
            if('online' in name):
                # assert np.max(result_list[:args.timestamp])<0.3
                result_list=result_list[args.timestamp:]
            print("{} count of {}, with mean of {}".format(name,len(result_list), np.mean(result_list)))
        else:
            result_list=np.array(result_list)
            if('online' in name):
                continue
                # assert np.max(result_list[:args.timestamp])<0.3
                index_list=get_online_protocol_index(class_=args.timestamp)
            else:
                index_list=get_offline_protocol_index(class_=args.timestamp)
            if(args.verbose==1):
                print(result_list)
            result_list=[str(np.mean(result_list[np.array(item[1])])) for item in index_list.items()]
            key_list=[item[0] for item in index_list.items()]
            print("{} with {} of {}".format(name,", ".join(key_list),", ".join(result_list)))
    if(args.move==1):
        print('finish moving all data from local node to the main server!')

