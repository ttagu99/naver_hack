import os
import sys
import time
import subprocess
from multiprocessing import Process
from datetime import datetime

###########################################################################################
team_name = "Zonber"       
data_name = "ir_ph2"     
sessions = ['427','428','429']  
models = ['over_over_fitting']
#models = ['secls_222_29','secls_222_30','secls_222_31','secls_222_32','secls_222_33','secls_222_34','secls_222_35','secls_222_36']        
start_wait_sec =  0#40*60         
###########################################################################################


def run_submit(command):
    now = time.time()
    print(f"[Command] {command}")
    print(datetime.now())
    subprocess.call(command)
    print(f"[Collapsed time] {time.time() - now}")


if __name__ == "__main__":
    S_HOUR = 3601
    li_procs = []

    for s, m in zip(sessions, models):
        full_session = '/'.join([team_name, data_name, s])
        full_command = f"nsml submit {full_session} {m}"
        li_procs.append(Process(target=run_submit, args=(full_command, )))

    now = time.time()
    time.sleep(start_wait_sec)

    for i, proc in enumerate(li_procs):
        proc.start()
        if i + 1 < len(li_procs):
            time.sleep(S_HOUR)

    for proc in li_procs:
        proc.join()
    print(f"Total collapsed time: {time.time() - now}")