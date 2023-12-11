import nidaqmx
from nidaqmx import constants, stream_readers, stream_writers
import scipy.signal as signal

import time
import numpy as np
import matplotlib.pyplot as plt
import string as str
from datetime import datetime
import os


class MemProgrammer:


    def __init__(self, device_name: str = "myDAQ1",fs_acq: float = 10000, N: int = 1e4,r = 4.7, states_limit: tuple = (4, 50)):

        self.task_write = nidaqmx.Task(new_task_name='Writing task')
        self.task_read = nidaqmx.Task(new_task_name="Reading task")
        self.task_read.ai_channels.add_ai_voltage_chan(f"{device_name}/ai0","channel1",
                                                        constants.TerminalConfiguration.DIFF,
                                                        -5, 5,)
        self.task_read.ai_channels.add_ai_voltage_chan(f"{device_name}/ai1", "channel2", constants.TerminalConfiguration.DIFF, -5, 5)
        self.task_write.ao_channels.add_ao_voltage_chan(f"{device_name}/ao0", 'write_channel', -3, 3)

        self.task_read.timing.cfg_samp_clk_timing(rate=fs_acq, samps_per_chan=N,sample_mode=constants.AcquisitionType.FINITE)  # you may not need samps_per_chan

        self.task_write.timing.cfg_samp_clk_timing(rate=fs_acq, samps_per_chan=N, sample_mode=constants.AcquisitionType.FINITE,
                                                    active_edge=constants.Edge.FALLING)
        samples_per_buffer = int(fs_acq // 1)
        self.task_read.in_stream.input_buf_size = samples_per_buffer * 10  # plus some extra space
        self.task_write.out_stream.output_buf_size = samples_per_buffer * 10
        self.reader = stream_readers.AnalogMultiChannelReader(self.task_read.in_stream)
        self.writer = stream_writers.AnalogMultiChannelWriter(self.task_write.out_stream, auto_start=False)
        self.fs_acq = fs_acq
        self.N = N
        self.r = r
        self.limit = states_limit
  
    def __make_pulse(self,amp, dt):
        """
        :param N: number of samples
        :param amp: pulse amplitude
        :param dt: width of the pulse
        :param fs: sampling frequency
        :return: output vector
        """
        ts = 1 / self.fs_acq
        temp = self.N / self.fs_acq

        t_i = np.linspace(-temp / 2, temp / 2, self.N)
        x = amp * (np.heaviside(t_i + dt / 2, 0.5) - np.heaviside(t_i - dt / 2, 0.5))
        """
        plt.plot(x)
        plt.show()
        """
        return x.reshape((1, self.N))

    def __writing_and_reading(self, amp=0.15, dt=0.01):
        # sourcery skip: avoid-builtin-shadow
        N = self.N
        r = self.r
        self.writer.write_many_sample(self.__make_pulse(amp, dt), timeout=100)
        buffer = np.zeros((2, N), dtype=np.float64)

        self.task_write.start()
        self.reader.read_many_sample(buffer, N, timeout=-1)

        self.task_write.wait_until_done()
        self.task_write.stop()
        U = buffer[0,:]
        U_r = buffer[1,:]
        I = buffer[1,:] / r
        filter_m = np.logical_not(np.logical_and(U < 0.03, U > -0.03))
        filter_r = np.logical_not(np.logical_and(U_r < 0.03, U_r > -0.03))
        
        if len(U_r[filter_r]) > len(U[filter_m]):
            filter = filter_r
            U = U[filter]
            I = I[filter]
            U[0] = U[1]
            I[0] = I[1]
            U[-1] = U[-2]
            I[-1] = I[-2]
        else:
            filter = filter_m
            U = U[filter]
            I = I[filter]
            U[-1] = U[-2]
            I[-1] = I[-2]
            U[0] = U[1]
            I[0] = I[1]

        U_f = signal.savgol_filter(U, 31, 3)
        I_f = signal.savgol_filter(I, 31, 3)

        '''
        plt.figure(2)
        plt.title("Setting")
        plt.plot(U_f,label='Napięcie')
        plt.plot(I_f, label='Prąd')
        plt.legend()
        plt.show()
        '''

        return U, I, U_f, I_f



    def __check_resistane(self, amp=0.15, dt=0.01):
        ''' Check if the resistance is within a certain range.'''
        print("Checking resistance")

        U,I,U_f,I_f = self.__writing_and_reading(amp=amp, dt=dt)

        """
        plt.figure(1)
        plt.title("Checking")
        f, (ax1, ax2) = plt.subplots(2,1)
        ax1.plot(I[(np.logical_not(np.logical_and(U < 0.02, U > -0.02)))])
        ax1.plot(I_f)
        ax2.plot(U)
        ax2.plot(I)
        plt.show()
        """

        R = U/I

        R_f = U_f / I_f
        R = R[R > 0]
        R_f = R_f[R_f > 0]
        print(f"R={np.mean(R)}, R_f={np.mean(R_f)}")

        return np.mean(R_f)




    def __set_Ron_state(self, amp: float=1.0, dt: float=0.01, saving=False):
        """ Setting memristor in R_on state"""
        print(f'Setting R_on state with parameters Amp={amp}, dt={dt}')

        U,I,U_f,I_f = self.__writing_and_reading(amp=amp, dt=dt)

        #peaks, _ = signal.find_peaks(I, height=np.mean(I), width=dt * N)
        #print(peaks)
        fs = self.fs_acq
        
        t_i = np.arange(start=0, step=1 / fs, stop=len(I_f) * 1 / fs, dtype=np.float64)
        q = np.trapz(x=t_i, y=I_f)
        # print(I_f)
        # print(U_f)
        p_i = np.multiply(U_f, I_f)
        E = np.trapz(x=t_i, y=p_i)
        print(f"q={q},\t E={E}")
        """
        plt.figure(2)
        plt.title("Setting")
        plt.plot(U)
        plt.plot(I)
        plt.show()
        """
  

        if saving:
            self.__loging(t_i,U,I,f"logs_setting_Ron_state_Amp={amp}, dt={dt}")
        return q, E



    def __loging(self,t_i,U,I,directory):
        """AI is creating summary for __loging

        Args:
            t_i ([type]): [description]
            U ([type]): [description]
            I ([type]): [description]
            directory ([type]): [description]
        """

        # sourcery skip: use-fstring-for-concatenation

        if os.path.exists(directory) == False:
            os.mkdir(directory)

        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
        work_file = date_time+'.csv'
        file_path = os.path.join(directory,work_file)
        concat = np.concatenate((t_i.reshape(len(t_i),1),
                                    U.reshape(len(t_i),1),
                                    I.reshape(len(t_i),1)),axis=1)
        np.savetxt(file_path, concat, delimiter=',')




    def __set_Roff_state(self,dt=0.1, amp=-1.5,saving=False):
        """ Setting memristor in R_off state"""
        
        print(f'Setting memristor in R_off state with parameters Amp={amp}, dt={dt}')
        
        U,I,U_f,I_f = self.__writing_and_reading(amp=amp, dt=dt)

        #peaks, _ = signal.find_peaks(I, height=np.mean(I), width=dt * N)
        #print(peaks)
        fs = self.fs_acq

        t_i = np.arange(start=0, step=1 / fs, stop=len(I_f) * 1 / fs, dtype=np.float64)
        q = np.trapz(x=t_i, y=I_f)
        # print(I_f)
        # print(U_f)
        p_i = np.multiply(U_f, I_f)
        E = np.trapz(x=t_i, y=p_i)
        print(f"q={q},\t E={E}")
        """
        plt.figure(2)
        plt.title("Setting")
        plt.plot(U)
        plt.plot(I)
        plt.plot(peaks, I[peaks], "x")
        plt.title("Resetting")
        plt.show()
        """

        if saving:
            self.__loging(t_i,U,I,f"logs_setting_Roff_state_Amp={amp}, dt={dt}")
        return q, E


    def __check_state(self,r):
        """ Checking current state of memristor"""
        if r <= self.limit[0]:
            return "R_on"
        elif r >= self.limit[1]:
            return "R_off"
        else:
            return "Unknown"


    def setting_Ron_measurment(self, n_mem,Amp_On, Amp_Off, dt_On, dt_Off,max_tests, max_pulses, saving=False, directory="Programowanie_Ron_wyniki"):

        fs_acq = self.fs_acq  # sample frequency
        N = self.N  # number of samples
        # dt = 0.5e-2
        if  not os.path.exists(directory):
            os.mkdir(directory)

        file = f"Programowanie_Ron_wyniki_AmpOn={Amp_On}_dtOn={dt_On}_memNumber{n_mem}.csv"
        file_path = os.path.join(directory,file)

        string = "Timestamp,No. pulses, No. Test,R,Succes,dt_Ron,Amp_RonR,q,E_memristor,State\n"

        with open(file_path, "w") as f:
            f.write(string)


        q,E = self.__set_Roff_state(dt=dt_Off, amp=Amp_Off,saving=saving)
        R = self.__check_resistane()
        state = self.__check_state(R)

        tests = 0
        pulses = 0
        succ = False
        
        desired_state = "R_on" if state == "R_off" else "R_off"
        string = f"{time.time()}, {pulses}, {tests}, {R},{succ}, {dt_Off}, {Amp_Off}, {q},{E},{state}\n"
        while tests <= max_tests-1:

            with open(file_path, "a") as f:
                f.write(string)

            if desired_state == "R_off" or pulses >= max_pulses:
                q,E = self.__set_Roff_state(dt=dt_Off, amp=Amp_Off,saving=saving)
                pulses = 0
                succ = False
                R = self.__check_resistane()
                state = self.__check_state(R)
                string = f"{time.time()}, {pulses}, {tests}, {R},{succ}, {dt_Off}, {Amp_Off}, {q},{E},{state}\n"
                print(f"Oczekiwany stan: {desired_state} - Otrzymany stan: {state} , R={R}, puls nr: {pulses}")
                if state == "R_off":
                    desired_state = "R_on"
                    tests += 1
                else:
                    desired_state = "R_off"
                    _, _ = self.__set_Ron_state(dt=0.1, amp=1.5,saving=saving)
                    
                        # zeroing(task_ write)
            else:
                q, E = self.__set_Ron_state(dt=dt_On, amp=Amp_On,saving=saving)
                pulses = pulses + 1
                R = self.__check_resistane()
                state = self.__check_state(R)
                
                
                
                if  desired_state == state:
                    succ = True
                else:
                    succ = False
                string = f"{time.time()}, {pulses}, {tests}, {R},{succ}, {dt_On}, {Amp_On}, {q},{E},{state}\n"
                print(f"Oczekiwany stan: {desired_state} - Otrzymany stan: {state} , R={R}, puls nr: {pulses}")
                if state == "R_on":
                        desired_state = "R_off"
                else:
                        desired_state = "R_on"
                        
                
                # zeroing(task_write)

            
            # zeroing(task_write)

           


    def closing(self):
        """ Closing the connection to the device"""
        self.task_read.stop()
        self.task_write.stop()
        self.task_read.close()
        self.task_write.close()
        print("Connection closed")
        
        
    def setting_Roff_measurment(self, n_mem,Amp_On, Amp_Off, dt_On, dt_Off,max_tests, max_pulses, saving=False, directory="Programowanie_Roff_wyniki"):

        fs_acq = self.fs_acq  # sample frequency
        N = self.N  # number of samples
        # dt = 0.5e-2
        if  not os.path.exists(directory):
            os.mkdir(directory)

        file = f"Programowanie_Roff_wyniki_AmpOff={Amp_Off}_dtOff={dt_Off}_memNumber{n_mem}.csv"
        file_path = os.path.join(directory,file)

        string = "Timestamp,No. pulses,No. Test,R,Succes,dt_Ron,Amp_RonR,q,E_memristor,State\n"

        with open(file_path, "w") as f:
            f.write(string)


        q,E = self.__set_Ron_state(dt=dt_On, amp=Amp_On,saving=saving)
        R = self.__check_resistane()
        state = self.__check_state(R)

        tests = 0
        pulses = 0
        succ = False
        with open(file_path, "a") as f:
            while tests <= max_tests:
                
                desired_state = "R_on" if state == "R_off" else "R_off"
                
                if desired_state == "R_off":
                    string = f"{time.time()}, {pulses}, {tests}, {R},{succ}, {dt_Off}, {Amp_Off}, {q},{E},{state}\n"
                else:
                    string = f"{time.time()}, {pulses}, {tests}, {R},{succ}, {dt_On}, {Amp_On}, {q},{E},{state}\n"
                
                f.write(string)

                if desired_state == "R_on" or pulses >= max_pulses:
                    q,E = self.__set_Ron_state(dt=dt_On, amp=Amp_On,saving=saving)
                    tests += 1
                    pulses = 0

                            # zeroing(task_ write)
                else:
                    q, E = self.__set_Roff_state(dt=dt_Off, amp=Amp_Off,saving=saving)
                    pulses = pulses + 1
                    # zeroing(task_write)

                R = self.__check_resistane()
                state = self.__check_state(R)
                # zeroing(task_write)

                print(f"Oczekiwany stan: {desired_state} - Otrzymany stan: {state} , R={R}")
                if desired_state == "R_off" and desired_state == state:
                    succ = True
                else:
                    succ = False

    def setting_Ron_once(self,dt_On=0.1, Amp_On: float= 1.5,saving=False):
        q,E = self.__set_Ron_state(dt=dt_On, amp=Amp_On ,saving=saving)
        R = self.__check_resistane()
        state = self.__check_state(R)
        return R, state
    
    def check_resistane(self, amp=0.15, dt=0.01):
        ''' Check if the resistance is within a certain range.'''
        return self.__check_resistane(amp=amp, dt=dt)

    def __spinwait_us(self,delay):
        target = time.perf_counter_ns() + delay * 1e9
        while time.perf_counter_ns() < target:
            pass

    def check_retention(self, n_mem, delays=30, time_of_test=3600):
        R = self.__check_resistane(amp=0.15, dt=0.01)
        directory = "Retentions_test"

        if os.path.exists(directory) == False:
            os.mkdir(directory)

        file  = f"mem{n_mem}_retention_test.csv"
        file_path = os.path.join(directory,file)
        string = "time,R\n"

        with open(file_path, "w") as f:
            f.write(string)

        while R >= 2:
            R, state =  self.setting_Ron_once(dt_On=0.1, Amp_On=1.5,saving=False)
        print(f"R ={R} kOhm is set ")

        start = time.perf_counter_ns()
        target = time.perf_counter_ns() + time_of_test * 1e9

        with open(file_path, "a") as fp:
            while time.perf_counter_ns() <= target:
                R = self.__check_resistane(amp=0.15, dt=0.01)
                string = f"{(time.perf_counter_ns()-start)/1e9}, {R}\n"
                fp.write(string)
                self.__spinwait_us(delay=delays)

    def __check_3_state(self,r, ranges: tuple):
        """ Checking current state of memristor"""
        if r <= ranges[0]:
            return 0
        elif       ranges[1][0] <= r <=  ranges[1][1]:
            return 1
        elif r >=  ranges[2]:
            return 2
        else:
            return np.NAN
        
    def set_3_states(self, desired_state, dts: iter, amps: iter, ranges) -> bool:
        R = self.__check_resistane()
        state = self.__check_3_state(R,ranges)
        count = 0
        
        while state != desired_state:
            if desired_state == 0:
                q,E = self.__set_Ron_state(dt=dts[0], amp=amps[0] ,saving=False)
            elif desired_state == 1:
                q,E = self.__set_Ron_state(dt=dts[1], amp=amps[1] ,saving=False)
            else:
                q,E = self.__set_Roff_state(dt=dts[2], amp=amps[2] ,saving=False)
            
            R = self.__check_resistane()
            state = self.__check_3_state(R,ranges)
            
            count += 1
            print(f'Count = {count}')
            
            if desired_state != state:
                q,E = self.__set_Roff_state(dt=dts[2], amp=amps[2] ,saving=False)
        
        return True