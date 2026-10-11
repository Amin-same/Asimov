import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import psutil

plt.ion()
parent_pid = os.getppid()
fig, ax = plt.subplots(figsize=(12, 6))

last_change = time.time()

while True:

    try:
        if not psutil.pid_exists(parent_pid):
            print("Training-Prozess beendet")
            break
        mtime = os.path.getmtime("reward_debug.csv")
        if mtime > last_change:
            last_change = mtime

        df = pd.read_csv("reward_debug.csv")
        latest_episode = df["episode"].max()
        df = df[df["episode"] == latest_episode]
        ax.clear()

        meta_columns = {"episode","step"}
        for col in df.columns:
            if col in meta_columns:
                continue
            ax.plot(df["step"], df[col], label=col)

        ax.legend()
        ax.grid(True)

        plt.pause(0.1)

    except Exception as e:
        print(e)

    time.sleep(0.2)
