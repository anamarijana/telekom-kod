import random
import numpy as np
import math
import matplotlib.pyplot as plt
from bitstring import BitArray


def codeByHamming(generated_sign_whole, Nsim, n, k):
    
    #kodiranje signala preko Hammingovog koda

    generated_sign_part = [0]*k
    m = 0
    i = 0
    coded_sign_whole = []

    #cepamo signal na delove od po k

    while i < Nsim:
        while m < k:
            generated_sign_part[m] = generated_sign_whole[i]
            m = m + 1
            i = i + 1

        coded_sign_part = hammingEncode(n,k,generated_sign_part)
        m = 0
        coded_sign_whole.extend(coded_sign_part)

    return coded_sign_whole

def decodeByHamming(coded_sign_whole,Nsim,n,k):
    
    #dekodiranje signala preko Hammingovog koda

    coded_sign_part = [0]*n
    m = 0
    i = 0
    recieved_sign_whole = []
    new_size = (Nsim/k)*n 
    while i < new_size:  
        while m < n:
            coded_sign_part[m] = coded_sign_whole[i]
            m = m + 1
            i = i + 1
        

        recieved_sign_part = hammingDecode(n,k,coded_sign_part)
        m = 0
        recieved_sign_whole.extend(recieved_sign_part)

    return recieved_sign_whole

    
def repetitionEncode(generated_sign_whole, order):

    coded_sign_whole = []
    
    for i in range(0, len(generated_sign_whole)):
        for j in range(0, order):
            coded_sign_whole.append(generated_sign_whole[i])
        
    return coded_sign_whole

def repetitionDecode(coded_sign_whole, order):
    
    recieved_sign_whole = []
    
    for i in range(0, int(len(coded_sign_whole)/order)):
        sum = 0
        for j in range(0, order):
            sum = sum + coded_sign_whole[i*order + j]
            
        if sum >= int(order/2) + 1:
            recieved_sign_whole.append(1)
        else:
            recieved_sign_whole.append(0)
        
    return recieved_sign_whole

 
def submatrix (n, k):

  #pravi matricu binarnih predstava onih brojeva koji nisu stepen dvojke
 
    parity_num = n - k
    A = []
    m = 0
    #izdvajamo one brojeve cije se binarne predstave ne nalaze u jedinicnoj matrici
    for i in range(1, n + 1):
        p = math.pow(2, m)
        if i == p:
            m = m + 1
        else:
            A.append(i) 
    A.sort(reverse = True)
    B = np.zeros((k, parity_num))

       
    for i in range(0, k):
        for j in range(0, parity_num):
            parity = int(math.pow(2,j)) & int(A[i])
            if parity:
                B[i,parity_num-j-1] = 1
            else:
                B[i,parity_num-j-1] = 0
        
    return B

def errBurst(coded_sign_whole, burst_len, burst_freq):
    
    transfer_sign_whole = coded_sign_whole
    sign_len = len(coded_sign_whole)
    random_err_position = []
    
    #ovaj pristup nece raditi ako pokusamo da naguramo previse gresaka u previse kratak signal
    #nas signal ima 10000 bitova od cega ce greska biti u nekoliko desetina pa nece biti problema
    #ideja je da generisemo polozaje gresaka jednu po jednu
    #ako je doslo do preklapanja burst gresaka, regenerisemo poslednji polozaj dok ne otklonimo preklapanje
    for i in range(0, burst_freq):
        pos = random.randint(0, sign_len - 1)
        good = True
        if pos >= sign_len - burst_len - 1:
            good = False
        for old_pos in random_err_position:
            if pos >= old_pos and pos <= old_pos + burst_len:
                good = False
        while not good:
            good = True
            pos = random.randint(0, sign_len - 1)
            if pos >= sign_len - burst_len - 1:
                good = False
            for old_pos in random_err_position:
                if pos >= old_pos and pos < old_pos + burst_len + 1:
                    good = False
        random_err_position.append(pos)
                    
    print(random_err_position)

    
    for i in range(len(random_err_position)):
        for j in range (0,burst_len):
            transfer_sign_whole[random_err_position[i]+j] = not coded_sign_whole[random_err_position[i]+j]
            
    return transfer_sign_whole
    
    
def errProbability(coded_sign_whole, err_prob):

    #kreiranje greske odredjene verovatnocom iste
    #kreiranje signala sa greskom
   
    transfer_sign_whole = np.zeros(len(coded_sign_whole))

    num_err_bits = 0
    index_err_bits = [] 
    
    for i in range(len(coded_sign_whole)):
       m= random.random()   
       if m <= err_prob:
           num_err_bits = num_err_bits + 1
           transfer_sign_whole[i] = not coded_sign_whole[i]
           
           index_err_bits.append(i)
       else:
           transfer_sign_whole[i] = coded_sign_whole[i]
    
    print("Broj gresaka: " + str(len(index_err_bits)))
    
    return transfer_sign_whole

def generator(Nsim, P0):

    generated_sign_whole = np.zeros(Nsim)
    for i in range(Nsim):
        m= random.random()   
        if m <= P0:
            generated_sign_whole[i] = 0
        else:
            generated_sign_whole[i] = 1
        
    return generated_sign_whole

def hammingEncode(n, k, generated_sign_part):

    B = submatrix (n, k)
    
    #Generatorska matrica
    Identity = np.identity(k) 
    G = np.concatenate((Identity, B), axis=1)
    #ovaj deo je potreban da bi indeksi bili redom rasporedjeni
    if(n==12):
         p8 = G[:,8]
         G = np.delete(G, 8, 1)
         G = np.insert(G, 4, p8, axis=1)
         G[:,[8, 9]] = G[:,[9, 8]]
    if n==7:
        G[:,[3, 4]] = G[:,[4, 3]]


    # kodiranje poruke
    result = np.matmul(generated_sign_part, G)
    coded_sign_part = result % 2
    return coded_sign_part

def hammingDecode(n,k,coded_sign_part):
    
    B = submatrix(n,k)
    H = np.concatenate((B, np.identity(n-k))) #Parity-check matrica

    # p8 d7 d6 d5 p4 d3 p2 p1
    # zbog oblika matrice kada je koji
    
    if n==12:
         p8 = H[8]
         H = np.delete(H, 8, 0)
         H = np.insert(H, 4, p8, axis=0)
         H[[8, 9]] = H[[9, 8]]
    if n==7:
        H[[3, 4]] = H[[4, 3]]

    syndrome = np.matmul(coded_sign_part, H)%2;                    
    
    position = BitArray(syndrome)
    error_pos = position.uint

    if error_pos:
        coded_sign_part[n-(error_pos)] =  1 - coded_sign_part[n-(error_pos)]
   
    coded_sign_part = np.flip(coded_sign_part, 0)
   
    recived_sign_part = []
    m = 0
    
    #izdvajamo one brojeve cije se binarne predstave ne nalaze u jedinicnoj matrici
    #jer su oni deo originalnog signala
    
    for i in range(1, n + 1):
        p = math.pow(2, m)
        if i==p:
            m = m + 1
        else:
            recived_sign_part.append(coded_sign_part[i - 1]) 
        
    recived_sign_part = np.flip(recived_sign_part, 0)
        
    return recived_sign_part

def interleave(input_sequence, word_length, no_of_words):

    output_sequence = [0] * len(input_sequence)
    
    if len(input_sequence) % word_length != 0:
        return -1
   
    #remaining_words = int(float(len(input_sequence)) / word_length) % no_of_words # da li ce biti neka rec van matrice
    no_of_passes = int(float(int(float(len(input_sequence))) / word_length) / no_of_words) #broj matrica
    
    for rep in range (0,no_of_passes):
        iN = [0] * (word_length*no_of_words) 
        ouT = [0] * (word_length*no_of_words)
    
        #interleaving matrix
        matrix =np.zeros((no_of_words, word_length))
        
        #construct word block to interleave
        for pos in range(0, len(iN)):
            iN[pos] = input_sequence[rep*word_length*no_of_words + pos]
       
        
        #construct matrix in row-first order
        for ct1 in range (0, no_of_words):
            for ct2 in range(0, word_length):
                matrix[ct1, ct2] = iN[ct1*word_length + ct2]
         
   
        
        #read matrix in column first order
        for ct1 in range(0, word_length):
            for ct2 in range (0, no_of_words):
                ouT[ct1*no_of_words + ct2] = matrix[ct2, ct1]
           
       
        
        #copy word block to output sequence
        for pos in range(0, len(ouT)):
            output_sequence[rep*word_length*no_of_words + pos] = ouT[pos]
       
   
    
    
    return output_sequence

def simulate(Nsim, P0, err_prob, burst, burst_len, burst_freq, interleaving, no_of_words):
    #originalni signal
    generated_sign_whole = generator(Nsim, P0)
        
    #kodiranje
    H7coded_sign_whole = codeByHamming(generated_sign_whole, Nsim, 7, 4)
    H12coded_sign_whole = codeByHamming(generated_sign_whole, Nsim, 12, 8)
    R3coded_sign_whole = repetitionEncode(generated_sign_whole, 3)
    R5coded_sign_whole = repetitionEncode(generated_sign_whole, 5)
    R7coded_sign_whole = repetitionEncode(generated_sign_whole, 7)
        
    #interliving
    if interleaving:
        H7coded_sign_whole = interleave(H7coded_sign_whole, 7, no_of_words)
        H12coded_sign_whole = interleave(H12coded_sign_whole, 12, no_of_words)
        R3coded_sign_whole = interleave(R3coded_sign_whole, 3, no_of_words)
        R5coded_sign_whole = interleave(R5coded_sign_whole, 5, no_of_words)
        R7coded_sign_whole = interleave(R7coded_sign_whole, 7, no_of_words)
        
    #greska pri prenosu
    if not burst:
        H7coded_sign_whole = errProbability(H7coded_sign_whole, err_prob)
        H12coded_sign_whole = errProbability(H12coded_sign_whole, err_prob)
        R3coded_sign_whole = errProbability(R3coded_sign_whole, err_prob)
        R5coded_sign_whole = errProbability(R5coded_sign_whole, err_prob)
        R7coded_sign_whole = errProbability(R7coded_sign_whole, err_prob)
    else:
        H7coded_sign_whole = errBurst(H7coded_sign_whole, burst_len, burst_freq)
        H12coded_sign_whole = errBurst(H12coded_sign_whole, burst_len, burst_freq)
        R3coded_sign_whole = errBurst(R3coded_sign_whole, burst_len, burst_freq)
        R5coded_sign_whole = errBurst(R5coded_sign_whole, burst_len, burst_freq)
        R7coded_sign_whole = errBurst(R7coded_sign_whole, burst_len, burst_freq)
        
    #deinterliving
    if interleaving:
        H7coded_sign_whole = interleave(H7coded_sign_whole, no_of_words, 7)
        H12coded_sign_whole = interleave(H12coded_sign_whole, no_of_words, 12)
        R3coded_sign_whole = interleave(R3coded_sign_whole, no_of_words, 3)
        R5coded_sign_whole = interleave(R5coded_sign_whole, no_of_words, 5)
        R7coded_sign_whole = interleave(R7coded_sign_whole, no_of_words, 7)
        
    #dekodiranje
    H7recieved_sign_whole = decodeByHamming(H7coded_sign_whole, Nsim, 7, 4)
    H12recieved_sign_whole = decodeByHamming(H12coded_sign_whole, Nsim, 12, 8)
    R3recieved_sign_whole = repetitionDecode(R3coded_sign_whole, 3)
    R5recieved_sign_whole = repetitionDecode(R5coded_sign_whole, 5)
    R7recieved_sign_whole = repetitionDecode(R7coded_sign_whole, 7)
    
    #parametri simulacije
    print("P0: " + str(P0))
    print("Burst: " + str(burst))
    if burst:
        print("Duzina burst greske: " + str(burst_len))
        print("Broj burst greska: " + str(burst_freq))
    else:
        print("Verovatnoca greske: " + str(err_prob))
        print("Ocekivan broj gresaka: " + str(int(Nsim*err_prob)))
    print("Interleaving: " + str(interleaving))
    if interleaving:
        print("Broj reci za interliving: " + str(no_of_words))
        
    #uporedjivanje izlaznih signala sa ulaznim
    H7BER = 0
    H12BER = 0
    R3BER = 0
    R5BER = 0
    R7BER = 0
    for i in range(0, Nsim):
        if H7recieved_sign_whole[i] != generated_sign_whole[i]:
            H7BER = H7BER + 1
        if H12recieved_sign_whole[i] != generated_sign_whole[i]:
            H12BER = H12BER + 1
        if R3recieved_sign_whole[i] != generated_sign_whole[i]:
            R3BER = R3BER + 1
        if R5recieved_sign_whole[i] != generated_sign_whole[i]:
            R5BER = R5BER + 1
        if R7recieved_sign_whole[i] != generated_sign_whole[i]:
            R7BER = R7BER + 1
            
    print("STOPE GRESAKA PO KODU");
    print("Hamming (7,4): " + str(H7BER) + " neispravljenih gresaka")
    print("Hamming (12,8): " + str(H12BER) + "neispravljenih gresaka")
    print("Repeticioni kod 3 ponavljanja: " + str(R3BER) + "neispravljenih gresaka")
    print("Repeticioni kod 5 ponavljanja: " + str(R5BER) + "neispravljenih gresaka")
    print("Repeticioni kod 7 ponavljanja: " + str(R7BER) + "neispravljenih gresaka")
    
    return [R3BER, R5BER, R7BER, H7BER, H12BER]

def plot_errors(error_list, title, fname):
    
    plt.style.use('ggplot')
    
    x = ['Rep3', 'Rep5', 'Rep7', 'Hamm(7,4)', 'Hamm(12,8)']
    
    x_pos = [i for i, _ in enumerate(x)]
    
    plt.bar(x_pos, error_list, color='green')
    plt.xlabel("Zaštitni kod")
    plt.ylabel("Broj neispravljenih grešaka (manje je bolje)")
    plt.title(title)
    
    plt.xticks(x_pos, x)
    
    plt.savefig(fname)

def main():
    
    #upotreba:
    #simulate(Nsim, P0, err_prob, burst, burst_len, burst_freq, interleaving, no_of_words)
    #plot(error_list, title, fname)
    
    Nsim = 10003
    P0 = 0.5
    err_prob = 0.01
    
    #slučaj pojedinačne greške
    #interleave = [5, 7, 9]
    #single_error = simulate(Nsim, P0, err_prob, False, 0, 0, True,5 )
    #plot_errors(single_error, "Pojedinačne greške verovatnoće " + str(0.01), "single.png")
    
    #burst greska bez interleavinga
    burst_len = [4, 5, 8, 10]
    burst_freq = [10, 8, 5, 4]
    for i in range(0, 4):
        burst_error = simulate(Nsim, P0, err_prob, True, burst_len[i], burst_freq[i], False, 0)
        plot_errors(burst_error, str(burst_freq[i]) + " burst grešaka dužine " + str(burst_len[i]) + " bez interlivinga", "burst_" + str(burst_len[i]) + ".png")
        
    #burst greska sa interleavingom
    interleave = [5, 7, 9]
    for i in range(0, 4):
        for j in range(0, 3):
            burst_error = simulate(Nsim, P0, err_prob, True, burst_len[i], burst_freq[i], True, 7)
            plot_errors(burst_error, str(burst_freq[i]) + " burst grešaka dužine " + str(burst_len[i]) + " sa interlivingom " + str(interleave[j]) + " reči", "burst_" + str(burst_len[i]) + "_int_" + str(interleave[j]) + ".png")
    
    
if __name__ == "__main__":
    main()
