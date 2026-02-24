from avalanche.training.plugins import StrategyPlugin
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from torch.optim import SGD, AdamW
from torchvision.models import resnet18, resnet50, resnet101
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torchvision
from transformers import get_cosine_schedule_with_warmup
from typing import Sequence
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, \
    loss_metrics, timing_metrics, cpu_usage_metrics, confusion_matrix_metrics, disk_usage_metrics
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.gdumb import GDumbPlugin
from avalanche.training.strategies.strategy_wrappers import SAIDO
from avalanche.training.strategies import Naive, CWRStar, Replay, GDumb, Cumulative, GEM, AGEM, EWC, JointTraining, \
    SynapticIntelligence, CoPE, OWM, RAWM, RWM, RegO
from avalanche.training.strategies.icarl import ICaRL
from avalanche.training.strategies.ar1 import AR1
from avalanche.training.strategies.deep_slda import StreamingLDA
from avalanche.training.plugins.early_stopping import EarlyStoppingPlugin
from avalanche.training.plugins.load_best import LoadBestPlugin
from avalanche.models.SAIDO_CLIP import SAIDO_MultiScene
from load_dataset import *

import argparse
from get_config import *
#from extract_feature import *
from parse_log_to_result import *
import glob
import json
import random
from datetime import datetime
from avalanche.models.SCNN import CNNSelfAttention as scnn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_logger(name):
    # log to text file with timestamp to avoid overwriting
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = './{}/log/log_{}_{}.txt'.format(args.split, name, timestamp)
    text_logger = TextLogger(open(log_filename, 'w'))

    # print to stdout
    interactive_logger = InteractiveLogger()
    tb_logger = TensorboardLogger('./{}/tb_data'.format(args.split))
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, epoch_running=True, experience=True, stream=True,
                         trained_experience=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        # timing_metrics(epoch=True, epoch_running=True),
        forgetting_metrics(experience=True, stream=True),
        # cpu_usage_metrics(experience=True),
        confusion_matrix_metrics(num_classes=args.num_classes, save_image=False, stream=True),
        # disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger, text_logger, tb_logger]
    )
    return text_logger, interactive_logger, eval_plugin


def make_scheduler(optimizer, step_size, gamma=0.1):
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma
    )
    return scheduler

def get_all_scene_ids(txt_path: str):
    """Collect all scene_ids from annotation file (return deduplicated string list)"""
    scene_ids = set()
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            scene_id = parts[2]  # Third field
            scene_ids.add(str(scene_id))  # Ensure converted to string
    return sorted(scene_ids)

global args
args = get_config()
set_seed(args.seed)

try:
    restart = int(args.restart)
except:
    print('restart flag must be 0/1')
    assert False
if (restart == 1):
    print('enter Y/y to continue')
    value = input()
    if (value == "y" or value == 'Y'):
        assert False
        print('remove old split folder')

os.makedirs("./{}".format(args.split), exist_ok=True)
os.makedirs("./{}/log/".format(args.split), exist_ok=True)
os.makedirs("./{}/model/".format(args.split), exist_ok=True)
os.makedirs("./{}/metric/".format(args.split), exist_ok=True)
method_query = args.method.split()  # list of CL method to run

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.cuda.get_device_name(0)
# torch.cuda.device_count()
print(device)

'''
Remember to delete the old feature path before generating new feature 
'''

for strate in method_query:
    for current_mode in ['offline']:
        # skip previous train model if necessary
        import glob

        model_path = sorted(glob.glob('../{}/model/model_{}_{}*'.format(args.split, strate, current_mode)))
        if (len(model_path) == 0 and args.eval == True):
            checkpoint_path = './{}/model/model_{}_{}*'.format(args.split, strate, current_mode)
            print('Checkpoint for model {} is not found at path {}'.format(strate, checkpoint_path))
            continue
        if (len(model_path) != 0):
            model_path = model_path[-1]
            state_dict = torch.load(model_path)
        else:
            state_dict = None
        if (current_mode == 'offline'):
            scenario = get_lora_scene_scenario(args)
        else:
            print("error")
        print('========================================================')
        print('========================================================')
        print('current strate is {} {}'.format(strate, current_mode))
        print('========================================================')
        print('========================================================')
        if args.pretrain_feature == 'None':
            model = SAIDO_MultiScene(last_k=0)
        else:
            model = nn.Linear(args.pretrain_feature_shape, args.num_classes)
        train_list_path = args.train_txt  # Assume train.txt path is defined in config
        with open(train_list_path, "r") as f:
            all_lines = f.readlines()
        data_count = len(all_lines)
        all_scene_ids = get_all_scene_ids(train_list_path)
        print("All scene labels:", all_scene_ids)
        model.init_all_scenes(all_scene_ids)
        print('data_count (from txt) is {}'.format(data_count))
        data_count = min(args.max_memory_size, data_count)
        if (strate.split("_")[-1].isnumeric() == False):
            buffer_size = data_count
        else:
            buffer_size = int(strate.split("_")[-1])

        if torch.cuda.device_count() > 1:
            print("Let's use all GPUs!")
            model = nn.DataParallel(model)
        else:
            print("only use one GPU")
        if (args.load_prev == True and state_dict is not None):
            result = model.load_state_dict(state_dict, strict=False)
            print()
            print('loaded previous model {}'.format(model_path))
            # Only output when there are exceptions
            if result.missing_keys or result.unexpected_keys:
                print('Warning: Parameters do not fully match')
                if result.missing_keys:
                    print(f'  Missing keys: {len(result.missing_keys)} parameters')
                    if len(result.missing_keys) <= 3:
                        for key in result.missing_keys:
                            print(f'    - {key}')
                    else:
                        for key in result.missing_keys[:2]:
                            print(f'    - {key}')
                        print(f'    ... and {len(result.missing_keys) - 2} more parameters')
                if result.unexpected_keys:
                    print(f'  Unexpected keys: {len(result.unexpected_keys)} parameters')
                    if len(result.unexpected_keys) <= 3:
                        for key in result.unexpected_keys:
                            print(f'    - {key}')
                    else:
                        for key in result.unexpected_keys[:2]:
                            print(f'    - {key}')
                        print(f'    ... and {len(result.unexpected_keys) - 2} more parameters')
            print()
        if (torch.cuda.is_available()):
            model = model.cuda()
        optimizer = SGD(list(filter(lambda x: x.requires_grad, model.parameters())), lr=args.start_lr,
                        weight_decay=float(args.weight_decay), momentum=args.momentum)
        scheduler = make_scheduler(optimizer, args.step_schedular_decay, args.schedular_step)

        plugin_list = [LRSchedulerPlugin(scheduler), LoadBestPlugin('train_stream')]
        text_logger, interactive_logger, eval_plugin = build_logger("{}_{}".format(strate, current_mode))
        if strate == 'CWRStar':
            cl_strategy = CWRStar(
                model, optimizer,
                CrossEntropyLoss(), cwr_layer_name=None, train_mb_size=args.batch_size, train_epochs=args.nepoch,
                eval_mb_size=args.batch_size,
                evaluator=eval_plugin, device=device, plugins=plugin_list)
        elif 'Replay' in strate:
            cl_strategy = Replay(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch,
                eval_mb_size=args.batch_size, mem_size=buffer_size,
                evaluator=eval_plugin, device=device, plugins=plugin_list)
        elif (strate == 'JointTraining' and current_mode == 'offline'):
            cl_strategy = JointTraining(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch * args.timestamp // 3,
                eval_mb_size=args.batch_size,
                evaluator=eval_plugin, device=device, plugins=plugin_list)
        elif 'GDumbFinetune' in strate:
            cl_strategy = GDumb(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch,
                eval_mb_size=args.batch_size,
                evaluator=eval_plugin, device=device, plugins=plugin_list, mem_size=buffer_size, reset=False,
                buffer='class_balance')
        # stanard gdumb= reset model+ class_balance buffer'
        elif 'GDumb' in strate:
            cl_strategy = GDumb(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch,
                eval_mb_size=args.batch_size,
                evaluator=eval_plugin, device=device, plugins=plugin_list, mem_size=buffer_size, reset=True,
                buffer='class_balance')
        elif 'BiasReservoir' in strate:
            if ('reset' in strate):
                resett = True
            else:
                resett = False
            alpha_mode = 'Dynamic' if 'Dynamic' in strate else 'Fixed'
            alpha_value = float(strate.split("_")[-1])
            cl_strategy = GDumb(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch,
                eval_mb_size=args.batch_size,
                evaluator=eval_plugin, device=device, plugins=plugin_list, mem_size=buffer_size, reset=resett,
                buffer='bias_reservoir_sampling',
                alpha_mode=alpha_mode, alpha_value=alpha_value)
        elif 'Reservoir' in strate:
            cl_strategy = GDumb(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch,
                eval_mb_size=args.batch_size,
                evaluator=eval_plugin, device=device, plugins=plugin_list, mem_size=buffer_size, reset=False,
                buffer='reservoir_sampling')
        elif 'Cumulative' in strate:
            if ('reset' in strate):
                resett = True
            else:
                resett = False
            cl_strategy = Cumulative(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch,
                eval_mb_size=args.batch_size,
                evaluator=eval_plugin, device=device, plugins=plugin_list, reset=resett)
        elif strate == 'OWM':
            cl_strategy = OWM(
                model, optimizer,
                CrossEntropyLoss(),
                alpha=np.linspace(0, 2, num=args.timestamp).tolist(), temperature=1,
                train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin, device=device, plugins=plugin_list)
        elif strate == 'RAWM':
            cl_strategy = RAWM(
                model, optimizer,
                CrossEntropyLoss(),
                alpha=np.linspace(0, 2, num=args.timestamp).tolist(), temperature=1,
                train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin, device=device, plugins=plugin_list)
        elif strate == 'RWM':
            cl_strategy = RWM(
                model, optimizer,
                CrossEntropyLoss(),
                alpha=np.linspace(0, 2, num=args.timestamp).tolist(), temperature=1,
                train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin, device=device, plugins=plugin_list)
        elif strate == 'RegO':
            cl_strategy = RegO(
                model, optimizer,
                CrossEntropyLoss(),
                alpha=np.linspace(0, 2, num=args.timestamp).tolist(), temperature=1,
                ef_thresh=0.1, importnat_thresh=0.75,
                train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin, device=device, plugins=plugin_list)
        elif strate == 'SAIDO':
            cl_strategy = SAIDO(
                model, optimizer,
                CrossEntropyLoss(),
                alpha=np.linspace(0, 2, num=args.timestamp).tolist(), temperature=1,
                ef_thresh=0.1, importnat_thresh=0.75,
                train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin, device=device, plugins=plugin_list)
        elif strate == 'GEM':
            cl_strategy = GEM(
                model, optimizer,
                CrossEntropyLoss(), patterns_per_exp=data_count, memory_strength=0.5, train_mb_size=args.batch_size,
                train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin, device=device, plugins=plugin_list)
        elif 'AGEMFixed' in strate:
            cl_strategy = AGEM(
                model, optimizer,
                CrossEntropyLoss(), patterns_per_exp=buffer_size, sample_size=buffer_size,
                train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin, device=device, plugins=plugin_list, reservoir=True)
        elif 'AGEM' in strate:
            cl_strategy = AGEM(
                model, optimizer,
                CrossEntropyLoss(), patterns_per_exp=buffer_size, sample_size=buffer_size,
                train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin, device=device, plugins=plugin_list, reservoir=False)
        elif strate == 'EWC':
            cl_strategy = EWC(
                model, optimizer,
                CrossEntropyLoss(), ewc_lambda=0.4, mode='online', decay_factor=0.1,
                train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin, device=device, plugins=plugin_list)
        elif strate == 'Naive':
            cl_strategy = Naive(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch,
                eval_mb_size=args.batch_size,
                evaluator=eval_plugin, device=device, plugins=plugin_list)
        elif strate == 'SynapticIntelligence':
            cl_strategy = SynapticIntelligence(
                model, optimizer,
                CrossEntropyLoss(), si_lambda=0.0001, train_mb_size=args.batch_size, train_epochs=args.nepoch,
                eval_mb_size=args.batch_size,
                evaluator=eval_plugin, device=device, plugins=plugin_list)
        elif 'CoPE' in strate:
            cl_strategy = CoPE(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch,
                eval_mb_size=args.batch_size, mem_size=buffer_size,
                evaluator=eval_plugin, device=device, plugins=plugin_list)
        else:
            continue
        
        cl_strategy.save_dir = f'./{args.split}/importance_logs'
        os.makedirs(cl_strategy.save_dir, exist_ok=True)
        
        print('Starting experiment...')
        train_metric = {}
        test_metric = {}
        if (strate == 'JointTraining' and current_mode == 'offline'):
            model_save_path = './{}/model/model_{}_{}_time{}.pth'.format(args.split, strate, current_mode, 0)
            # 选择验证流：优先使用 scenario.val_stream，否则回退到 test_stream
            val_streams = getattr(scenario, 'val_stream', None)
            if val_streams is None:
                val_streams = scenario.test_stream
            if (args.eval == False):
                train_metric[0] = cl_strategy.train(scenario.train_stream, val_streams=val_streams)
            test_metric[0] = cl_strategy.eval(scenario.test_stream)
            print('current strate is {} {}'.format(strate, current_mode))
            torch.save(model.state_dict(), model_save_path)
            with open("./{}/metric/train_metric_{}.json".format(args.split, strate), "w") as out_file:
                json.dump(train_metric, out_file, indent=6)
            with open("./{}/metric/test_metric_{}.json".format(args.split, strate), "w") as out_file:
                json.dump(test_metric, out_file, indent=6)

        else:
            train_list = scenario.train_stream
            cur_timestep = 0
            if (len(model_path) != 0 and args.load_prev == True):
                try:
                    with open("./{}/metric/train_metric_{}.json".format(args.split, strate), "r") as file:
                        prev_train_metric = json.load(file)
                    with open("./{}/metric/test_metric_{}.json".format(args.split, strate), "r") as file:
                        prev_test_metric = json.load(file)
                    # extract ../clear100_imgnet_res50/model/model_BiasReservoir_Dynamic_1.0_offline_time05.pth as 5
                    load_prev_time_index = int(model_path.split('_')[-1].split('.')[0][4:])
                    train_list = train_list[load_prev_time_index + 1:]
                    cur_timestep = load_prev_time_index + 1
                    test_metric = prev_test_metric
                    train_metric = prev_train_metric
                    print('start runing from bucket {}'.format(cur_timestep))
                except:
                    pass

            val_streams = getattr(scenario, 'val_stream', None)
            if val_streams is None:
                print('ERROR val_stream is None')
                val_streams = scenario.test_stream
            
            for experience in train_list:
                model_save_path = './{}/model/model_{}_{}_time{}.pth'.format(args.split, strate, current_mode,
                                                                              str(cur_timestep).zfill(2))
                print("Start of experience: ", experience.current_experience)
                print("Current Classes: ", experience.classes_in_this_experience)
                print('current strate is {} {}'.format(strate, current_mode))
                # offline
                if (current_mode == 'offline'):
                    # train returns a dictionary which contains all the metric values
                    print('current strate is {} {}'.format(strate, current_mode))
                    print('Training completed')
                    print('Computing accuracy on the whole test set')
                    
                    if (args.eval == False):
                        
                        current_val_stream = None
                        if isinstance(val_streams, Sequence) and len(val_streams) > cur_timestep:
                            current_val_stream = val_streams[cur_timestep]
                        elif not isinstance(val_streams, Sequence):
                            current_val_stream = val_streams
                        train_metric[cur_timestep] = cl_strategy.train(experience, val_streams=current_val_stream)
                    test_metric[cur_timestep] = cl_strategy.eval(scenario.test_stream)
                    print('current strate is {} {}'.format(strate, current_mode))
                # online
                else:
                    print('current strate is {} {}'.format(strate, current_mode))
                    print('Computing accuracy on the future timestamp')
                    test_metric[cur_timestep] = cl_strategy.eval(scenario.test_stream)
                    if (args.eval == False):
                        
                        current_val_stream = None
                        if isinstance(val_streams, Sequence) and len(val_streams) > cur_timestep:
                            current_val_stream = val_streams[cur_timestep]
                        elif not isinstance(val_streams, Sequence):
                            current_val_stream = val_streams
                        train_metric[cur_timestep] = cl_strategy.train(experience, val_streams=current_val_stream)
                    
                    print('Training completed')
                    print('current strate is {} {}'.format(strate, current_mode))
                torch.save(model.state_dict(), model_save_path)
                log_path = './{}/log/'.format(args.split)
                log_name = 'log_{}.txt'.format("{}_{}".format(strate, current_mode))
                with open("./{}/metric/train_metric_{}.json".format(args.split, strate), "w") as out_file:
                    json.dump(train_metric, out_file, indent=6)
                with open("./{}/metric/test_metric_{}.json".format(args.split, strate), "w") as out_file:
                    
                    test_metric[cur_timestep]['ConfusionMatrix_Stream/eval_phase/test_stream'] = \
                        test_metric[cur_timestep]['ConfusionMatrix_Stream/eval_phase/test_stream'].numpy().tolist()
                    json.dump(test_metric, out_file, indent=6)
                out_file.close()
                cur_timestep += 1
                if (args.eval == True):
                    break
