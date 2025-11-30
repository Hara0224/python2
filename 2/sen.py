import socket
#import smbus
import time
import datetime
import numpy as np
import spidev
import RPi.GPIO as gpio
import tkinter as tk
#from matplotlib import pyplot as plt

def func():
    global labelTime
    global labelCTemp
    global labelStatus
    global labeCK1615
    global label34410A_in
    global label34410A_out
    global labelVin
    global status
    
    labeltext=datetime.datetime.now()
    labelTime.config(text=str(labeltext)[:19],)

#現在の温度を測定する
    temperature = MeasTemp()
    labelCTemp.config(text=temperature)


#    SetTemp(StartTemp,4.7,1)



    if(temperature>=emergencyTemp):
        emergency_stop()
        exit()

#    if temperature > StartTemp : #現在気温が高いので冷却する
#        labelStatus.config(text='冷却中')
#        SetCooling(StartTemp,4.7)
    if(status=='冷却中'):
        SetTemp(StartTemp,4.7,1)
#        labelStatus.config(text='冷却終了')
#        status='冷却終了'


    app.after(100,func)

def emergency_stop():
    #非常停止
    #試験終了
    gpio.output(emergency,gpio.HIGH)
    time.sleep(0.5)
    gpio.output(emergency,gpio.LOW)
    print("非常停止しました")

def MeasTemp():
    readByteArray = spi.xfer2(dummyData)
    temperatureData = ((readByteArray[0] & 0b01111111) << 5) | ((readByteArray[1] & 0b11111000) >> 3)
    temperature = temperatureData * 0.25
    return temperature

def MeasSF(temp,TestVoltage):
    with open("/home/wakimoto/20240531_1.txt","a") as f:

        #waitを置く
        time.sleep(0.2)
        #VHを印加
        CK1615.send(b'SIG 1\n')
        #waitを置く
        time.sleep(0.2)
        for k in range(10):
            Vin.send(b'read?\n')
            Vout.send(b'read?\n')
            response_in=Vin.recv(1024)
            response_out=Vout.recv(1024)
            In = float(response_in.decode('utf8'))
            Out=float(response_out.decode('utf8'))
            SF=In/Out
            print(k,SF)
            f.write(str(temp)+" , "+str(k)+" , " +str(SF)+"\n")
    #            print(MeasSF)
    CK1615.send(b'SIG 0\n') #スタンバイ


    return
'''
def SetCooling(temperature,Vin):
#    global labelTime
#    global labelCTemp
    global labelStatus
#    global label6161
#    global label34460
#    global labelVin

#    print('冷却開始')
    status='冷却中'
    labelStatus.config(text='冷却中')
    time.sleep(0.5)


#極性反転
    gpio.output(polinv,gpio.HIGH)
    SetTemp(temperature,Vin,1)
#    print('冷却終了')
    labelStatus.config(text='冷却終了')
    time.sleep(0.5)
    return


def SetHeating(temperature,vin):
    global labelTime
    global labelCTemp
    global labelStatus
    global label6161
    global label34460
    global labelVin

#    print('加熱開始')
    labelStatus.config(text='加熱中')
    #極性通常
#    gpio.output(polinv,gpio.HIGH)
    gpio.output(polinv,gpio.LOW)

    SetTemp(temperature - 1.0,vin,1)
#    print('加熱終了')
    labelStatus.config(text='加熱終了')

    return
'''

def SetTemp(temperature,vin,mode):
    global labelTime
    global labelCTemp
    global labelStatus
    global labelCK1615
    global label34410A_in
    global label34410A_out
    global labelVin
    global labelSetTemp

    M=0.00
    M1=0.00
    goal = float(temperature)
    e=0.0
    e1=0.00
    e2=0.00
    Kp=0.1
    Ki=0.05
    Kd=0.9

    i=0
    while 1:
        temp1 = MeasTemp()
#    temperature = MeasTemp()
        labelCTemp.config(text=temp1)
#        print('TEMP1=',temp1)
        if(temp1>=emergencyTemp):#温度が高すぎたら強制停止
            emergency_stop()
            exit()
        currentTemp=float(temp1)
        if(mode==1):
            if (temp1<temperature):
                return
        elif(temp1==temperature):
                return
#        print(str(temp1) + ' ℃')
        if(temp1>temperature):
            gpio.output(polinv,gpio.HIGH)
        else:
            gpio.output(polinv,gpio.LOW)
        M1 = M
        e2 = e1
        e1 = e
        e = -(goal - currentTemp)
        M = M1 +(+ Kp * (e - e1) + Ki * e + Kd * ((e-e1)-(e1-e2)))
        
        M=(M*1.0)
        if(M>5):
            M=5.0
        if(M<0):
            M=5.0
        vin=vin+M
        #vin=abs(vin)
        if(vin)<0:
            vin=0.0
        if(vin>4.7):
            vin=4.7
            M=M*0.1
        #if(i==1000):
        print('設定温度：' + str(temperature) + ' ℃ ,'+ str(temp1) + ' ℃ , 温度差:' + str(e) + ', ドライブ電圧:'+ str(vin) + ' V, M:'+ str(M) )

        time.sleep(0.1)
        labelSetTemp.config(text=temperature)
        time.sleep(0.1)
        labelVin.config(text=str(vin)+" V")
        time.sleep(0.1)


#    i=0
        #電圧を出力させる
    #    vin=0.0
        #D/A変換へ入力するコード計算
        # TEXIOを使用するため、以下のI2C DAC関連のコードは不要になります。
        # setnum=int(vin/0.001221)
        # #print(e,M,vin,setnum)
        # msb=int(setnum/16)
        # lsb=(setnum-int(setnum/16)*16)*16
        # s = [msb,lsb]
        # i2c.write_i2c_block_data( MCP4725, 0x40 , s )
        #i+=1
    #return
    
def EqSet():
    global label34410A_in
    global label34410A_out
    global labelVin
    
    Vin.connect((VinHost, port)) #サーバーに接続
    Vout.connect((VoutHost,port))
    CK1615.connect((CK1615Host, 1234)) #サーバーに接続

    Vin.send(b'*idn?\n') #34410Aのid問い合わせ
    response = Vin.recv(4096) #データ受信
    print(response)

    Vout.send(b'*idn?\n') #34410Aのid問い合わせ
    response = Vout.recv(4096) #データ受信
    print(response)

    CK1615.send(b'++addr 2\n') #GBIBアドレス２
    CK1615.send(b'++mode 1\n') #デバイスモード

    CK1615.send(b'?IDT\n') #CK1615のid問合せ
    response = CK1615.recv(4096) #データ受信
    print(response)
    CK1615.send(b'SIG 0\n') #スタンバイ
    CK1615.send(b'HIV 2\n')
    CK1615.send(b'LOV -2\n')
    CK1615.send(b'FRQ 1000\n') #1000Vレンジ
    labelCK1615.config(text="ON")
    label34410A_in.config(text="ON")
    label34410A_out.config(text="ON")

    # ここからTEXIOのコードを追加
    TEXIOHost = "192.168.1.102" # <-- ここにTEXIOのIPアドレスを設定してください
    TEXIOPort = 2268 # TEXIOが使用するポート番号を設定してください。

    try:
        TEXIO.connect((TEXIOHost, TEXIOPort)) # TEXIOに接続
        TEXIO.send(b'*idn?\n') # TEXIOのID問い合わせコマンド（例）
        response_texio = TEXIO.recv(4096) # データ受信
        print(response_texio)
        # 必要に応じて、TEXIOの初期設定コマンドを追加
        # 例: TEXIO.send(b'VOLT 5.0\n') # 電圧設定コマンド
        # 例: TEXIO.send(b'OUTP OFF\n') # 初期状態では出力をOFFにしておく
        print("TEXIOに接続しました。")
    except Exception as e:
        print(f"TEXIOへの接続に失敗しました: {e}")
        # GUIに接続失敗を示すラベルがあれば、ここで更新
        # 例: labelTEXIO.config(text="OFF (Error)")


def TestStart():
    global labelTime
    global labelCTemp
    global labelStatus
    global labelCK1615
    global label34410A_in
    global label34410A_out
    global labelVin
    global buttonStart
    global status
    
    #試験開始
    '''
    gpio.output(teststart,gpio.HIGH)
    time.sleep(0.5)
    gpio.output(teststart,gpio.LOW)
    gpio.output(polinv,gpio.LOW)
    '''

    buttonStart.config(text="試 験 中")
    
    #現在の温度を測定する
    temperature = MeasTemp()
    labelCTemp.config(text=temperature)
    if(temperature>=emergencyTemp):
        emergency_stop()
        exit()

    if temperature > StartTemp : #現在気温が高いので冷却する
        labelStatus.config(text="冷却中")
        status='冷却中'
        time.sleep(0.5)
        gpio.output(polinv,gpio.HIGH)
        # ここにTEXIOの電圧いじるプログラム（冷却時）
        try:
            TEXIO.send(b'VOLT 0.0\n') # 冷却時のTEXIO電圧（例: 0V、または冷却に必要な電圧）
            TEXIO.send(b'OUTP ON\n') # 出力ON
            print("TEXIOを冷却モードに設定しました。")
        except Exception as e:
            print(f"TEXIO冷却設定エラー: {e}")
            
#        SetTemp(StartTemp,4.7,1)
#        labelStatus.config(text='冷却終了')
#        status='冷却終了'
        time.sleep(0.5)
    elif  temperature < StartTemp - 1.5: #現在気温が低いので加熱する
        labelStatus.config(text="加熱中")
        status='加熱中'
        time.sleep(0.5)
        #極性通常
        gpio.output(polinv,gpio.LOW)

        # ここにTEXIOの電圧いじるプログラム（加熱時）
        try:
            heating_voltage = 3.0 # 加熱に必要な電圧（例）
            TEXIO.send(f'VOLT {heating_voltage:.1f}\n'.encode()) # 加熱時のTEXIO電圧
            TEXIO.send(b'OUTP ON\n') # 出力ON
            print(f"TEXIOを加熱モードに設定しました。電圧: {heating_voltage}V")
        except Exception as e:
            print(f"TEXIO加熱設定エラー: {e}")

        #SetTemp(temperature - 1.0,vin,1)
        labelStatus.config(text='加熱終了')
        status='加熱終了'
        time.sleep(0.5)

'''
    for i in range(StartTemp, EndTemp+1 ,StepTemp):
        #現在の温度を測る
        temperature = MeasTemp()
        print('設定温度'+ str(i) + '℃' +'現在温度：' + str(temperature) + '℃' )
        for j in range(1):
    #    for j in range(19):
            SetTemp(float(i),1,-1)
            temperature = MeasTemp()
            print(str(j)+':設定温度'+ str(i) + '℃' +'現在温度：' + str(temperature) + '℃' )
            #if(j==19):
            MeasSF(temperature, TestVoltage)
            time.sleep(1)
    #        time.sleep(60)
    '''


def TestStop():
#試験中止
    global labelStatus
    global buttonStart
    #gpio.output(teststart,gpio.LOW)
    gpio.output(polinv,gpio.LOW)
    gpio.output(emergency,gpio.HIGH)
    
    # TEXIOの出力をオフにするコマンド
    try:
        TEXIO.send(b'OUTP OFF\n') # TEXIOの出力をオフにするコマンド
        print("TEXIOの出力を停止しました。")
        # ここにさらにTEXIOのコードを追加 (例: リセットコマンド)
        # TEXIO.send(b'*RST\n') # *RST はSCPI共通コマンドで、設定をリセットします。
        # print("TEXIOをリセットしました。")
    except Exception as e:
        print(f"TEXIO停止エラー: {e}")

    labelStatus.config(text="停止中")
    buttonStart.config(text="試験開始")

# I2C関連の初期化をコメントアウト
# i2c=smbus.SMBus(1)
# MCP4725=0x60

t=100
#GPIOの初期設定
polinv=19
#teststart=20
emergency=21
gpio.setmode(gpio.BCM)
gpio.setup(polinv,gpio.OUT)    # 黃線
#gpio.setup(teststart,gpio.OUT) # 赤線
gpio.setup(emergency,gpio.OUT) # 茶色

# SPIの初期設定
dummyData = [0x00,0x00]
spi = spidev.SpiDev()
spi = spidev.SpiDev()
spi.open(0, 0)          # bus 0,cs 0
spi.max_speed_hz = 1000000      # 1MHz
spi.mode = 3                    # SPI mode : 3


#DMM,電源の初期設定
VinHost = "192.168.1.100"  #34410Aのin　
VoutHost="192.168.1.101"   #34410Aのout
CK1615Host = "192.168.1.99"    #CK1615
TEXIO="192.168.1.102" # ここは空文字列ではなく、ソケットオブジェクトで初期化する必要があります。

port = 2268 

Vin = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #VHオブジェクトの作成
Vout = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
CK1615 = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #CK1615オブジェクトの作成
TEXIO=socket.socket(socket.AF_INET,socket.SOCK_STREAM) # ここでsocketオブジェクトを初期化

# I2C DACへの初期設定をコメントアウト
#s=[0xe8,0x80]
#i2c.write_i2c_block_data(MCP4725,0x40,s)

#開始温度
StartTemp = 0
#終了温度
EndTemp = 40
#温度ステップ
StepTemp = 1
#非常停止する温度
emergencyTemp = 70

#DAC電圧出力をVin[V]にする
vin=1.0
# I2C DAC関連のコードをコメントアウト
#setnum=int(vin/0.001221)
#print(e,M,vin,setnum)
#msb=int(setnum/16)
#lsb=(setnum-int(setnum/16)*16)*16
#s = [msb,lsb]
#i2c.write_i2c_block_data( MCP4725, 0x40 , s )
#s=[0x00,0x00]
#i2c.write_i2c_block_data(MCP4725,0x40,s)
gpio.output(emergency,gpio.LOW)

app=tk.Tk()
app.geometry("300x220")
label0=tk.Label(app,width=0,text="現在時刻")
label0.place(x=0,y=1)
labelTime=tk.Label(app)
labelTime.place(x=70,y=1)

label1=tk.Label(app,width=0,text="設定温度")
label1.place(x=0,y=60)
labelSetTemp =tk.Label(app,width=0,text=str(StartTemp))
labelSetTemp.place(x=70,y=60)

label2=tk.Label(app,width=0,text="℃　現在温度")
label2.place(x=110,y=60)
label2_5=tk.Label(app,width=0,text="℃")
label2_5.place(x=260,y=60)
labelCTemp=tk.Label(app,text=MeasTemp())
labelCTemp.place(x=210,y=60)

label3=tk.Label(app,width=0,text="Status")
label3.place(x=0,y=80)
status='停止中'
labelStatus=tk.Label(app,text="停止中")
labelStatus.place(x=70,y=80)

label4=tk.Label(app,width=0,text="CK1615")
label4.place(x=0,y=130)
labelCK1615=tk.Label(app,width=0,text="ON") # EqSetでONになるため、初期値もONが適切かもしれません。
labelCK1615.place(x=70,y=130)

label5=tk.Label(app,width=0,text="34410A_in")
label5.place(x=0,y=150)
label34410A_in=tk.Label(app,width=0,text="OFF")
label34410A_in.place(x=80,y=150)

label6=tk.Label(app,width=0,text="34410A_out")
label6.place(x=0,y=170)
label34410A_out=tk.Label(app,width=0,text="OFF")
label34410A_out.place(x=80,y=170)

label7=tk.Label(app,width=0,text="制御電圧")
label7.place(x=140,y=80)
labelVin=tk.Label(app,width=0,text="0.0 V")
labelVin.place(x=210,y=80)

label8=tk.Label(app,width=0,text="開始")
label8.place(x=0,y=20)
label9=tk.Label(app,width=0,text="℃　終了")
label9.place(x=90,y=20)
label10=tk.Label(app,width=0,text="℃ step")
label10.place(x=190,y=20)
label11=tk.Label(app,width=0,text="℃")
label11.place(x=270,y=20)
labelStartTemp=tk.Label(app,width=0,text=StartTemp)
labelStartTemp.place(x=50,y=20)
labelEndTemp=tk.Label(app,width=0,text=EndTemp)
labelEndTemp.place(x=160,y=20)
labelStepTemp=tk.Label(app,width=0,text=StepTemp)
labelStepTemp.place(x=250,y=20)

label12=tk.Label(app,width=0,text="非常停止温度")
label12.place(x=140,y=150)
label13=tk.Label(app,width=0,text="℃")
label13.place(x=270,y=150)
labelEmergencyTemp=tk.Label(app,width=0,text=emergencyTemp)
labelEmergencyTemp.place(x=240,y=150)

#機器接続のボタン
buttonEq=tk.Button(app,text="測定器接続",command=EqSet)
buttonEq.place(x=0,y=190)
#試験開始のボタン
buttonStart=tk.Button(app,text="試験開始",command=TestStart)
buttonStart.place(x=140,y=190)
#試験中止のボタン
buttonStop=tk.Button(app,text="試験中止",command=TestStop)
buttonStop.place(x=220,y=190)

app.after(100,func)
app.mainloop()
    
# プログラム終了時にソケットを閉じる処理
Vin.close()
Vout.close()
CK1615.close()
TEXIO.close() # TEXIOソケットも閉じる
gpio.cleanup()
