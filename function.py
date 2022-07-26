import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.io.matlab.mio5_params import mat_struct


'''
'''

## Fungsi-fungsi yang berkaitan dengan direktori
def make_subj_files(record_dir='recordings/',subj_codes=[],sess_codes=[]):
    '''
    membuat list berisi seluruh file dalam direktori
    input:
        record_dir : alamat direktori utama
        subj_codes : list berisi kode untuk setiap subjek
        sess_codes : list berisi kode untuk setiap sesi
    output:
        subj_files : dictionary berisi struktur folder
    '''
    subj_files = {'main_dir':record_dir}
    subj_dict = {}
    for folder_name in os.listdir(record_dir):
        subject,session,day = folder_info(folder_name)
        sess_day = session+'-'+f'{day:02d}'
        if subject not in subj_codes:
            subj_codes.append(subject)
        if session not in sess_codes:
            sess_codes.append(session)
        file_list = []
        for filename in os.listdir(record_dir+folder_name):
            file_list.append(filename)
        if bool(file_list):
            if subject not in subj_dict:
                subj_dict[subject] = {}
            if sess_day not in subj_dict[subject]:
                subj_dict[subject][sess_day] = []
            subj_dict[subject][sess_day] += file_list   
    subj_files['subjects'] = subj_codes
    subj_files['sessions'] = sess_codes
    subj_files['subj_dir'] = subj_dict 
    return subj_files
        
def folder_info(folder_name,prefix='OpenBCISession_',delim='-'):
    '''
    menampilkan informasi dari sebuah nama folder
    input:
        folder_name : nama folder yang akan dilihat informasinya
        prefix      : awalan dari nama folder
        delim       : jeda antar setiap informasi
    output:
        tuple berisi:
        subject     : kode subjek
        session     : kode sesi
        day         : nomor hari
    '''
    part = folder_name.replace(prefix,'')
    part = part.split(delim)
    subject = part[0]
    session = part[1]
    day = int(part[2])
    return (subject,session,day)

def print_filepath(subj_files=None,subject='',session='S1',day=0,trial=0,prefix='OpenBCISession_'):
    '''
    menampilkan alamat lengkap untuk sebuah trial
    input:
        subj_files : dictionary berisi struktur folder
        subject    : kode subjek
        session    : kode sesi
        day        : nomor hari
        trial      : nomor perulangan pada satu sesi per hari
        prefix     : awalan dari nama folder
    output
        filepath   : alamat lengkap dari file yang dilihat
    '''
    if subj_files == None:
        print("berkas subj_files harus dibuat terlebih dahulu.")
        return False
    
    filepath = subj_files['main_dir']
    filepath += prefix+subject+'-'+session+'-'+f'{day:02d}'+'/'
    filepath += subj_files['subj_dir'][subject][session+'-'+f'{day:02d}'][trial]
    return filepath

def read_file(subj_files=None,subject='',session='S1',day=0,trial=0,filepath='',prefix='OpenBCISession_'):
    '''
    membaca file dan menyimpannya dalam format panda frame
    input:
        subj_files : dictionary berisi struktur folder
        subject    : kode subjek
        session    : kode sesi
        day        : nomor hari
        trial      : nomor perulangan pada satu sesi per hari
        prefix     : awalan dari nama folder
    output
        data_pd    : data dalam format panda frame
        
    '''
    if subj_files == None:
        print("berkas subj_files harus dibuat terlebih dahulu.")
        return False
    
    if filepath == '':
        filepath = print_filepath(subj_files,subject,session,day,trial,prefix)
    
    emg_pd = pd.read_csv(filepath,header=4,sep=',')
    
    return emg_pd

def save_corrected(exg_pd_corrected,old_dir,new_dir,subj_files=None,subject='',session='S1',day=0,trial=0,
                   old_prefix='OpenBCISession_',new_prefix='OpenBCISession_'):
    '''
    menyimpan data exg terkoreksi ke dalam file txt berformat csv
    input:
        exg_pd_corrected : data dalam format panda frame yang terkoreksi
        old_dir          : folder utama rekaman exg sebelum dikoreksi
        new_dir          : folder utama rekaman exg setelah dikoreksi
        subj_files       : dictionary berisi struktur folder
        subject          : kode subjek
        session          : kode sesi
        day              : nomor hari
        trial            : nomor perulangan pada satu sesi per hari
        old_prefix       : awalan dari nama folder sesi sebelum dikoreksi
        new_prefix       : awalan dari nama folder sesi setelah dikoreksi
    output:
        boolean : status keberhasilan penyimpanan berkas
    '''
    file_info = '%OpenBCI Raw EEG Data\n'\
                '%Number of channels = 4\n'\
                '%Sample Rate = 1600 Hz\n'\
                '%Board = OpenBCI_GUI$BoardGanglionWifi\n'
    fp = print_filepath(subj_files=subj_files,subject=subject,session=session,day=day,trial=trial,prefix=old_prefix)
    if old_dir != new_dir:
        fp = fp.replace(old_dir,new_dir)
    if old_prefix != new_prefix:
        fp = fp.replace(old_prefix,new_prefix)
    fp = fp[:-4]+'_corrected.txt'
    if os.path.isfile(fp):
        print('berkas sudah ada:',fp)
        return False
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    part = fp.replace(new_dir,'').split('/')
    folder = part[0]
    if not os.path.isdir(new_dir+folder):
        os.mkdir(new_dir+folder+'/')
    
    with open(fp,'a+') as fo:
        fo.seek(0)
        fo.write(file_info)
    exg_pd_corrected.to_csv(fp,index=False,mode='a')
    print('berkas disimpan:',fp)
    return True

def correct_all_files(subj_files_rec,subj_files_corrected,correct_mark='auto',T_correct=[1.5,3]):
    '''
    mengoreksi titik awal data dari seluruh berkas
    input:
        subj_files_rec       : dictionary berisi struktur folder awal 
        subj_files_corrected : dictionary berisi struktur folder hasil koreksi
        correct_mark='auto'  : mode koreksi
            'auto'  : pencarian otomatis titik awal dengan fungsi 'correct_start()'
            integer : langsung dikoreksi ke titik awal tertentu (nomor indeks)
        T_correct            : rentang waktu pencarian, jika dengan mode 'auto'
    '''
    for subject, sess_day_dict_rec in subj_files_rec['subj_dir'].items():
        sess_day_dict = dict(sess_day_dict_rec.copy())
        for sess_day,file_list in sess_day_dict.items():
            part = sess_day.split('-')
            session = part[0]
            day = int(part[1])
            N_trial = len(file_list)
            for trial in range(N_trial):
                file_corrected = file_list[trial][:-4]+'_corrected.txt'
                if subject not in subj_files_corrected['subj_dir']:
                    subj_files_corrected['subj_dir'][subject] = {}
                if sess_day not in subj_files_corrected['subj_dir'][subject]:
                    subj_files_corrected['subj_dir'][subject][sess_day] = []
                if file_corrected not in subj_files_corrected['subj_dir'][subject][sess_day]:
                    emg_pd_incorrect = read_file(subj_files=subj_files_rec,
                                                    subject=subject,session=session,
                                                    day=day,trial=trial)
                    found = False
                    if correct_mark=='auto':
                        _,mark,found = correct_start(emg_pd_incorrect,T=T_correct,N_lsm=100,show=False)
                    elif isinstance(correct_mark,int):
                        mark = correct_mark
                        found = True
                    if found:
                        emg_pd = correct_exg_pd(emg_pd_incorrect,mark)
                    else:
                        emg_pd = emg_pd_incorrect
                        print('tidak ada koreksi:',subject,sess_day,file_corrected)
                    emg_record_dir = subj_files_rec['main_dir']
                    emg_corr_dir = subj_files_corrected['main_dir']
                    save_corrected(emg_pd,emg_record_dir,emg_corr_dir,subj_files=subj_files_rec,
                                      subject=subject,session=session,day=day,trial=trial)
                    subj_files_corrected['subj_dir'][subject][sess_day].append(file_corrected)
                else:
                    print('berkas sudah ada:',subject,sess_day,file_corrected)
    return True

## Fungsi-fungsi yang berkaitan dengan koreksi awalan data exg
def gradient_lsm(X,Y,N=0):
    '''
    menghitung gradien dari pasangan (X,Y) sepanjang N-titik
    input:
        X : data variabel x
        Y : data variabel y
    output:
        m : gradien atau slope dari regresi linier pasangan (X,Y)
    '''
    if N==0:
        N = len(X)
    m = (N*np.dot(X,Y)-np.sum(X)*np.sum(Y))
    m /= (N*np.dot(X,X)-(np.sum(X))**2)
    return m

def correct_start(exg_pd,T=[1.75,3.25],fs=1600,N_lsm=100,threshold=1e-3,show=False):
    '''
    mencari titik mulai data exg yang memiliki indeks waktu yang benar sesuai fs
    input:
        exg_pd    : data dalam format panda frame
        T         : rentang waktu pencarian
        fs        : frekuensi pencuplikan  
        N_lsm     : panjang data untuk perhitungan gradien
        threshold : batas toleransi gradien 
        show      : opsi menampilkan grafik
    output:
        gradients : daftar perubahan gradien sepanjang waktu
        mark      : indeks sebagai tanda titik koreksi
        found     : status ditemukannya titik koreksi
    '''
    ms_pd = pd.to_datetime(exg_pd.iloc[:,-1])
    ms_pd = ms_pd.dt.microsecond
    ms_np = ms_pd.to_numpy()
    ms_np = ms_np/1000000
    N = [int(ti*fs) for ti in T]
    X = np.arange(N[0],N[1])/fs
    Y = ms_np[N[0]:N[1]]

    gradients = []
    found = False
    mark = N[0]
    for i in range(N[1]-N[0]-N_lsm):
        m = gradient_lsm(X[i:i+N_lsm],Y[i:i+N_lsm],N_lsm)
        gradients.append(m)
        if (1-m)<threshold and not found:
            mark += i
            found = True
    
    if show:
        plt.figure(figsize=[12,8])
        plt.plot(X,Y)
        plt.plot(X[:-N_lsm],gradients)
        plt.plot([mark/fs,mark/fs],[0,1.1])
        plt.xlim(T)
        plt.ylim([-0.1,1.1])
        plt.legend(['time index','gradient','start index found'])
        plt.grid()
        
    return gradients,mark,found

def correct_exg_pd(exg_pd,mark=0):
    '''
    mengoreksi data dengan menghilangkan awalan data sebelum indeks "mark"
    input:
        exg_pd : data dalam format panda frame
        mark   : indeks dimulainya data exg yang baru
    output:
        exg_pd_corrected : data dalam format panda frame yang terkoreksi
    '''
    exg_pd_corrected = exg_pd.iloc[mark:,:]
    return exg_pd_corrected

def pd_to_numpy(exg_pd,board='Ganglion',sel_chan=False):
    '''
    mengkonversi data dalam format panda frame ke format numpy array
    input:
        emg_pd   : data dalam format panda frame
        board    : board yang digunakan 
        sel_chan : isi manual daftar channel yang ingin dikonversi
    output:
        emg_np   : data dalam format numpy array
            baris : channel
            kolom : data
    '''
    if not sel_chan:
        if board == 'Ganglion':
            sel_chan = [1,2,3,4]
        elif board == 'Cython':
            sel_chan = [1,2,3,4,5,6,7,8]
    emg_pd_chan = emg_pd.iloc[:,sel_chan]
    emg_np = emg_pd_chan.to_numpy().transpose()
    return emg_np  

def initiate_trial_starts(subj_files):
    '''
    Membuat dictionary kosong untuk menyimpan waktu mulai untuk setiap perekaman data exg
    input:
        subj_files   : dictionary berisi struktur folder
    output:
        trial_starts : dictionary berisi waktu mulai untuk setiap rekaman exg
    '''
    # trial_starts = dict(subj_files['subj_dir'].copy())
    trial_starts = {}
    for subject in subj_files['subj_dir'].keys():
        trial_starts[subject] = {}
        for sess_day,files in subj_files['subj_dir'][subject].items():
            N_trial = len(files)
            start_times = [0 for n in range(N_trial)]
            trial_starts[subject][sess_day] = start_times
            
    return trial_starts
    
def insert_trial_start(start_time=2.0,trial_starts=None,subject='',session='S1',day=0,trial=0,fs=0):
    '''
    Mengisi waktu mulai untuk setiap rekaman data exg
    input:
        start_index  : waktu mulai perekaman
        trial_starts : dictionary berisi waktu mulai untuk setiap rekaman exg
        subject      : kode subjek
        session      : kode sesi
        day          : nomor hari
        trial        : nomor perulangan pada satu sesi per hari
        fs           : frekuensi pencuplikan
            jika fs >  0, start_time adalah indeks data
            jika fs == 0, start_time adalah waktu (detik)
    output:
        trial_starts : dictionary berisi waktu mulai untuk setiap rekaman exg
    '''
    if trial_starts == None:
        print("berkas trial_time harus dibuat terlebih dahulu.")
        return False
    
    sess_day = session + '-' + f'{day:02d}'
    if subject not in trial_starts:
        print("subjek yang dipilih tidak ada.")
        return False
    elif sess_day not in trial_starts[subject]:
        print("sesi atau hari yang dipilih tidak ada.")
        return False
    
    N_trial = len(trial_starts[subject][sess_day])
    if trial >= N_trial:
        print("nomor trial yang dipilih tidak ada.")
        return False
    else:
        if fs > 0:
            start = start_time/fs
        else:
            start = start_time
        trial_starts[subject][sess_day][trial] = start
    
    return True

# Fungsi-fungsi yang terkait dengan grafik
def plot_marking_session(emg_np,sess_time=None,session='S1',time_start=0,channels=[0,1,2,3],fs=1600):
    '''
    menampilkan grafik emg dengan penanda untuk setiap pergantian gerakan
    input:
        emg_np     : data dalam format numpy array
        sess_time  : dictionary berisi gerakan beserta durasinya masing-masing
        session    : sesi yang dipilih
        time_start : waktu mulai perekaman gerakan pertama
        channels   : channels emg yang dipilih
        fs         : frekuensi pencuplikan
    '''
    if sess_time == None:
        print("dictionary sess_time harus dibuat terlebih dahulu.")
        return False
    
    sess_dur = sess_time[session]['dur']
    sess_act = sess_time[session]['act']
    
    N = emg_np.shape[1]
    tt = np.arange(N)/fs
    n_start = int(time_start*fs)
    t_start = time_start
    for ch in channels:
        emg_ch = emg_np[ch,:]
        max_emg = max(emg_ch)
        min_emg = min(emg_ch)
        plt.figure(figsize=[15,5])
        plt.plot(tt,emg_ch)
        t_now = t_start
        plt.plot([t_now,t_now],[min_emg,max_emg],'r')
        plt.grid()
        plt.title('Kanal {:d}'.format(ch))
        for i in range(len(sess_dur)):
            t_now += sess_dur[i]
            plt.plot([t_now,t_now],[min_emg,max_emg],'r')        
    return True

    

def matobj_to_dict(matobj):
    if isinstance(matobj,dict):
        return matobj
    dictobj = {}
    for key in matobj._fieldnames:
        val = matobj.__dict__[key]
        if isinstance(val,mat_struct):
            val = matobj_to_dict(val)
        dictobj[key] = val
    return dictobj

    #AKSES FILENAME
def file_info(file_name,postfix='.mat', delim='-'):
    '''
    menampilkan informasi dari sebuah nama folder
    input:
        file_name   : nama file yang akan dilihat informasinya
        delim       : jeda antar setiap informasi
    output:
        tuple berisi:
        subject     : kode subjek
        session     : kode sesi
        #thing       : kode objek
        day         : nomor hari
        trial       : kode trial
    '''
    part = file_name.replace(postfix,'')
    part = part.split(delim)
    subject = part[0]
    session = part[1]
    #thing   = part[2]
    day     = int(part[2])
    trial   = int(part[3])
    return (subject,session,day,trial)

def file_info1(file_name,postfix='.mat', delim='-'):
    '''
    menampilkan informasi dari sebuah nama folder
    input:
        file_name   : nama file yang akan dilihat informasinya
        delim       : jeda antar setiap informasi
    output:
        tuple berisi:
        subject     : kode subjek
        session     : kode sesi
        #thing       : kode objek
        day         : nomor hari
        trial       : kode trial
    '''
    part = file_name.replace(postfix,'')
    part = part.split(delim)
    subject = part[0]
    session = part[1]
    #thing   = part[2]
    day     = int(part[2])
    moves   = part[3]
    trial   = int(part[4])
    return (subject,session,day,moves,trial)


    # potong gerakan
folder = '../mat_files/1/'
fs = 1600


for file in os.listdir(folder):
    emg_filtered = loadmat(folder+file)
    emg = emg_filtered['emg_filtered']
    
    subject, session, day, trial = ob.file_info(file)
    
    if session == 'S1' or session == 'S2':
        sess_day = session+'-'+f'{day:02d}'
        ts = trial_starts[subject][sess_day][trial]
        ts_gs1 = ts + 8
        
        index = ts_gs1 * fs
        index_float = float(format(index, '2f'))
        index_start = int(index_float)
        index_stop = index_start +3200
        
        emg_gs1 = emg[:, index_start:index_stop]
        
        file = subject+'-'+sess_day+'-'+'S'+'-'+str(trial)+'.mat'
        filename = '../mat_files/2/'+file
        demg_gs1 = {'emg_movement': emg_gs1, 'label':'emg_gs1: emg gerakan supinasi1'}
        savemat(filename, demg_gs1)
        
    elif session == 'S3':
        sess_day = session+'-'+f'{day:02d}'
        ts = trial_starts[subject][sess_day][trial]
        ts_gs1 = ts + 13
        
        index = ts_gs1 * fs
        index_float = float(format(index, '2f'))
        index_start = int(index_float)
        index_stop = index_start +3200
        
        emg_gs1 = emg[:, index_start:index_stop]
        
        file = subject+'-'+sess_day+'-'+'S'+'-'+str(trial)+'.mat'
        filename = '../mat_files/2/'+file
        demg_gs1 = {'emg_movement': emg_gs1, 'label':'emg_gs1: emg gerakan supinasi1'}
        savemat(filename, demg_gs1)