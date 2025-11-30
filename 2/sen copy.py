# 改良版：TEXIOと測定器を用いた温度制御GUIシステム
import socket
import time
import datetime
import tkinter as tk
import spidev
import RPi.GPIO as gpio

# === 初期設定 ===
StartTemp = 0
EndTemp = 40
StepTemp = 1
emergencyTemp = 70
vin = 1.0
status = "停止中"
dummyData = [0x00, 0x00]

VinHost = "192.168.1.100"
VoutHost = "192.168.1.101"
CK1615Host = "192.168.1.99"
TEXIOHost = "192.168.1.102"
port = 2268

Vin = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
Vout = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
CK1615 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
TEXIO = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1000000
spi.mode = 3

polinv = 19
emergency = 21
gpio.setmode(gpio.BCM)
gpio.setup(polinv, gpio.OUT)
gpio.setup(emergency, gpio.OUT)
gpio.output(emergency, gpio.LOW)

# === 温度取得関数 ===
def MeasTemp():
    data = spi.xfer2(dummyData)
    value = ((data[0] & 0x7F) << 5) | ((data[1] & 0xF8) >> 3)
    return value * 0.25

# === 非常停止関数 ===
def emergency_stop():
    gpio.output(emergency, gpio.HIGH)
    time.sleep(0.5)
    gpio.output(emergency, gpio.LOW)
    print("[警告] 非常停止しました。")

# === PI制御で温度調整 ===
def SetTemp(goal, vin_init, mode):
    global status
    M, M1 = 0.0, 0.0
    e, e1, e2 = 0.0, 0.0, 0.0
    Kp, Ki, Kd = 0.1, 0.05, 0.9
    vin = vin_init
    
    while True:
        temp = MeasTemp()
        labelCTemp.config(text=f"{temp:.2f}")
        if temp >= emergencyTemp:
            emergency_stop()
            return
        if mode == 1 and temp < goal:
            return
        if abs(temp - goal) < 0.3:
            return

        gpio.output(polinv, gpio.HIGH if temp > goal else gpio.LOW)

        e2, e1 = e1, e
        e = -(goal - temp)
        M1 = M
        M = M1 + (Kp * (e - e1) + Ki * e + Kd * ((e - e1) - (e1 - e2)))
        M = max(0.0, min(M, 5.0))

        vin = max(0.0, min(vin_init + M, 4.7))

        try:
            TEXIO.send(f"VOLT {vin:.2f}\n".encode())
            TEXIO.send(b"OUTP ON\n")
        except:
            print("[エラー] TEXIO制御失敗")

        print(f"設定:{goal:.1f}℃ 測定:{temp:.2f}℃ 差:{e:.2f} 駆動電圧:{vin:.2f}V")
        labelVin.config(text=f"{vin:.2f} V")
        time.sleep(0.2)

# === 測定器初期化 ===
def EqSet():
    try:
        Vin.connect((VinHost, port))
        Vout.connect((VoutHost, port))
        CK1615.connect((CK1615Host, 1234))
        TEXIO.connect((TEXIOHost, port))

        for dev in [Vin, Vout]:
            dev.send(b"*idn?\n")
            print(dev.recv(4096))

        CK1615.send(b"++addr 2\n")
        CK1615.send(b"++mode 1\n")
        CK1615.send(b"?IDT\n")
        print(CK1615.recv(4096))

        TEXIO.send(b"*idn?\n")
        print(TEXIO.recv(4096))
        labelCK1615.config(text="ON")
        label34410A_in.config(text="ON")
        label34410A_out.config(text="ON")

    except Exception as e:
        print(f"[接続失敗] {e}")

# === 試験開始 ===
def TestStart():
    global status
    buttonStart.config(text="試 験 中")
    temp = MeasTemp()
    labelCTemp.config(text=f"{temp:.2f}")

    if temp >= emergencyTemp:
        emergency_stop()
        return

    if temp > StartTemp:
        labelStatus.config(text="冷却中")
        status = "冷却中"
        gpio.output(polinv, gpio.HIGH)
        SetTemp(StartTemp, 1.0, 1)
    elif temp < StartTemp - 1.5:
        labelStatus.config(text="加熱中")
        status = "加熱中"
        gpio.output(polinv, gpio.LOW)
        SetTemp(StartTemp, 1.0, 1)

    labelStatus.config(text="終了")
    status = "終了"
    buttonStart.config(text="試験開始")

# === 試験停止 ===
def TestStop():
    global status
    gpio.output(polinv, gpio.LOW)
    gpio.output(emergency, gpio.HIGH)
    try:
        TEXIO.send(b"OUTP OFF\n")
    except:
        pass
    labelStatus.config(text="停止中")
    buttonStart.config(text="試験開始")

# === GUI関連 ===
def update():
    try:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        labelTime.config(text=now)
        labelCTemp.config(text=f"{MeasTemp():.2f}")
    except:
        pass
    app.after(1000, update)

app = tk.Tk()
app.geometry("320x250")

labelTime = tk.Label(app)
labelTime.place(x=10, y=5)
labelSetTemp = tk.Label(app, text=str(StartTemp))
labelSetTemp.place(x=120, y=40)
labelCTemp = tk.Label(app, text="0.00")
labelCTemp.place(x=200, y=40)
labelStatus = tk.Label(app, text="停止中")
labelStatus.place(x=10, y=70)
labelCK1615 = tk.Label(app, text="OFF")
labelCK1615.place(x=10, y=100)
label34410A_in = tk.Label(app, text="OFF")
label34410A_in.place(x=10, y=120)
label34410A_out = tk.Label(app, text="OFF")
label34410A_out.place(x=10, y=140)
labelVin = tk.Label(app, text="0.0 V")
labelVin.place(x=10, y=160)

buttonEq = tk.Button(app, text="測定器接続", command=EqSet)
buttonEq.place(x=10, y=190)
buttonStart = tk.Button(app, text="試験開始", command=TestStart)
buttonStart.place(x=120, y=190)
buttonStop = tk.Button(app, text="試験中止", command=TestStop)
buttonStop.place(x=220, y=190)

app.after(1000, update)
app.mainloop()

# 終了処理
Vin.close()
Vout.close()
CK1615.close()
TEXIO.close()
gpio.cleanup()