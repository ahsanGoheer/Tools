# LookUp Table Class.
class LookUpTable(dict):
    def __init__(self):
        self=dict()
    def add(self,key,value):
        self[key]=value

# Hash Values For Corresponding Urdu Letters. 
hash = LookUpTable()
hash.add('0', '\u0627')  # ا
hash.add('1', '\u0622')  # آ
hash.add('2', '\u0628')  # ب
hash.add('4', '\u067E')  # پ
hash.add('5', '\u062A')  # ت
hash.add('6', '\u0679')  # ٹ
hash.add('7', '\u062B')  # ث
hash.add('8', '\u062C')  # ج
hash.add('9', '\u0686')  # چ
hash.add('A', '\u062D')  # ح
hash.add('B', '\u062E')  # خ
hash.add('C', '\u062F')  # د
hash.add('D', '\u0688')  # ڈ
hash.add('E', '\u0630')  # ذ
hash.add('F', '\u0631')  # ر
hash.add('G', '\u0691')  # ڑ  
hash.add('H', '\u0632')  # ز
hash.add('I', '\u0698')  # ژ
hash.add('K', '\u0633')  # س
hash.add('L', '\u0634')  # ش
hash.add('M', '\u0635')  # ص
hash.add('N', '\u0636')  # ض
hash.add('O', '\u0637')  # ط
hash.add('P', '\u0638')  # ظ
hash.add('Q', '\u0639')  # ع
hash.add('R', '\u063A')  # ‌غ 
hash.add('S', '\u0641')  # ف 
hash.add('T', '\u0642')  # ‌ق 
hash.add('U', '\u06A9')  # ‌ک
hash.add('V', '\u06AF')  # ‌گ
hash.add('W', '\u0644')  # ‌ل
hash.add('X', '\u0645')  # ‌م
hash.add('Z', '\u0646')  # ‌ن 
hash.add('Y', '\u06BA')  # ‌ں     
hash.add('a', '\u0648')  # ‌و 
hash.add('h', '\u06C1')  # ‌ہ
hash.add('3', '\u06BE')  # ھ
hash.add('b', '\u0621')  # ‌ء
hash.add('c', '\u06CC')  # ئ
hash.add('J', '\u06D2')  # ے
hash.add('f', ' ')       # space
hash.add('l', '\u06D4')  
hash.add('i', '\u060C')  # :
hash.add('g', '(') 
hash.add('j', ')') 
hash.add('m', '\u061F') 
hash.add('n', '\u0624') 
hash.add('\'', '\u06CC') #ی
hash.add('p', '\u0670') #ی khari zabar

def get_char(str):
    try:
        return hash[str]
    except:
        return hash['N']

def formate_urdu_text(UrduText):
   result=0
   for i in range(len(UrduText)):     
        pre='\0'
        post='\0'
        current=='\0'
        current = UrduText[i]
        if i!=0:
            pre=UrduText[i-1]
        if (i+1!=len(UrduText)):
            post = UrduText[i+1]
        if((pre!='\0' or post!='\0') and (current!=' ')):
            if(pre!='\0' and (post=='\0' or post ==' ')):
                if (pre != '\uFE8D' and pre != '\uFEAB' and pre != '\uFEA9' and pre != '\uFB8C' and pre != '\uFEAD' and pre != '\uFB8A' and pre != '\uFEAF' and pre != '\uFBAE' and pre != '\uFEED' and pre != '\uFB8A'):
                      current+=1
            elif(((pre == '\0' or pre == ' ') and post != '\0')):
                if ( current != '\uFE8D'  and  current != '\uFEAB'  and  current != '\uFEA9'  and  current != '\uFB8C'  and  current != '\uFEAD'  and  current != '\uFB8A'  and  current != '\uFEAF'  and  current != '\uFBAE' and  current != '\uFEED' and  current != '\uFB8B'):
                     current+=1 
                     current+=1
            else:
                if ( current == '\uFE8D'  or  current == '\uFEAB'  or  current == '\uFEA9'  or  current == '\uFB8C'  or  current == '\uFEAD'  or  current == '\uFB8A'  or  current == '\uFEAF'  or  current == '\uFBAE'  or  current == '\uFEED'):
                    if (pre != '\uFE8D' and pre != '\uFEAB' and pre != '\uFEA9' and pre != '\uFB8C' and pre != '\uFEAD' and pre != '\uFB8A' and pre != '\uFEAF' and pre != '\uFBAE' and pre != '\uFEED' and pre != '\uFB8A'):
                         current+=1
                    else:
                         current+=1
                         current+=1
                         current+=1
    
        result+=current
   return result


