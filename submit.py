import os
import sys
import time
import subprocess
from multiprocessing import Process
from datetime import datetime
import pandas as pd

team_name = "Zonber"       
data_name = "ir_ph2"     
start_wait_sec =  6*60#35*60
submit_list_path  = './submit_list.csv'

def run_submit(command):
    now = time.time()
    print(f"[Command] {command}")
    print(datetime.now())
    subprocess.call(command)
    print(f"[Collapsed time] {time.time() - now}")

if __name__ == "__main__":
    S_HOUR = 3601
    li_procs = []

    while(1):
        s_df = pd.read_csv(submit_list_path)
        for i in range(s_df.shape[0]):
            s = s_df.session[i]
            m = s_df.model[i]
            full_session = '/'.join([team_name, data_name, str(s)])
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
        s_df.drop(s_df.index, inplace=True)
        s_df.to_csv(submit_list_path)
