import numpy as np
import node as nde
class treetools:
    #初始化类，参数为 ：数据的属性集， 数据集， 数据的标签属性
    def __init__(self,attribute,data,label):
        self.attribute=attribute
        self.data=data
        self.label=label

    
    # 根据属性划分数据集
    def getattributeD(self,data,name):

        attNum=len(data)
        #属性所在的索引位置
        index=-1
        for j in range(len(self.attribute)):
             if (self.attribute)[j] == name:
                index=j

        #样本集合章所有样本在该属性上的所有取值；再去掉重复的
        attdata=data[:, index]
        attdata=np.unique(attdata)
        
        
        #根据属性的取值：创建字典 key为属性的取值 value 为根据取值化的数据集合
        Datadict={}
        datanum=[]  #存放样本在该属性上的取值
        for i in range(attNum):
            dt=data[i]   #取样本
            datanum.append(dt[index])  #存放样本在该属性上的取值
            for j in range(len(attdata)):  #根据样本在改属性上的取值划分数据集
                if dt[index]==attdata[j]:
                    try:
                        c=Datadict[attdata[j]]   #如果改属性在字典还不存在，抛出错误
                    except KeyError:
                        Datadict[attdata[j]] = dt  #字典中没有使，第一次创建
                    else:

                        Datadict[attdata[j]] =np.append(c,dt)   #字典存在该属性时，则取出数据，合并后在放入key对应的value

        # 把key 对应value 改成矩阵形式
        for i in Datadict:
            b = (Datadict[i]).reshape(int(Datadict[i].shape[0] / len(self.attribute)), len(self.attribute))
            Datadict[i]=b

        return Datadict,datanum


#按数值大小划分数据集
    def getattributeDByNumber(self, data,name, t):
        attNum = len(data)
        # 属性所在的索引位置
        index = -1
        for j in range(len(self.attribute)):
            if (self.attribute)[j] == name:
                index = j

        # 创建字典  把小于t 放入一个集合，把大于t的放入一个集合
        Datadict = {'-':[],'+':[]}
        for i in range(attNum):
            dt = data[i]
            if float(dt[index]) < t:
                    c = Datadict['-']
                    Datadict['-']=np.append(c, dt)
            else:
                    c = Datadict['+']
                    Datadict['+']=np.append(c, dt)

        for i in Datadict:
            b = (Datadict[i]).reshape(int(Datadict[i].shape[0] / len(self.attribute)), len(self.attribute))
            Datadict[i] = b

        return Datadict




    # 计算集合的信息熵
    def inforEntro(self,data):

        Data,datanum=self.getattributeD(data,self.label)
        pk=[]
        for i in Data:
            b=Data[i].shape[0]
            pk.append(b/data.shape[0])

        pk=np.array(pk)
        EntD=0;
        for i in range(len(pk)):
            EntD+=pk[i]*np.log2(pk[i])
        EntD=-EntD
        return EntD






    #计算某个属性的信息增益
    def inforGain(self,attrnum,data):
        #集合信息熵
        EntD=self.inforEntro(data)
        #根据属性取值划分集合
        DataV, datanumV = self.getattributeD(data, attrnum)
        Dnum=data.shape[0]
        EntDV=0
        #判断这个属性是不是连续值属性；不是连续性属性会抛出错误
        try :
            float(datanumV[0])
        except(ValueError):
            #计算信息增益
            for i in DataV:
                 DVnum=DataV[i].shape[0]
                 EntDV+=(DVnum/Dnum)*self.inforEntro(DataV[i])
            return EntD - EntDV,-1   #-1用来判断是不是连续属性
        else:
            #计算连续属性信息增益
            EntDV,EntT = self.NuminforGain(EntD,data,attrnum,datanumV)
            return EntDV,EntT





#处理连续属性的信息熵 ；找到信息增益最大的t

    def NuminforGain(self,EntD,data,attrnum,datanumV):
        for i in range(len(datanumV)):
            datanumV[i]=float( datanumV[i])
        datanumV=np.sort(datanumV)
        #在对连续属性的取值排序后，计算两两取值的平均数
        Ta=[]
        for i in range(len(datanumV)):
            if i + 1 != len(datanumV):
                Ta.append((datanumV[i]+datanumV[i+1])/2)
        Ta=np.array(Ta)


        #在Ta找到信息增益最大的t
        Dnum = data.shape[0]
        EntDV=0
        EntT=0
        for j in range(len(Ta)):
            DataV=self.getattributeDByNumber(data,attrnum , Ta[j])
            temp=0
            for q in DataV:
                DVnum = DataV[q].shape[0]
                temp += (DVnum / Dnum) * self.inforEntro(DataV[q])
            temp=EntD-temp
            if EntDV <temp:
                EntDV=temp
                EntT=Ta[j]

        # for i in range(len(Ta)):
        return EntDV,EntT



    #判断集合样本取值是否相同
    def eampleissample(self,data):
        isequal=True
        for i in range(data.shape[0]):
            for j in range(data.shape[0]):
                b=data[i]
                b=b[0:len(self.attribute)-1]
                c=data[j]
                c=c[0:len(self.attribute)-1]
                if (b==c).all() == False:
                    isequal=False

        return isequal


    #返回最多样本标签
    def backMostLabel(self,data):
        d,b=self.getattributeD(data,self.label)
        num1=0
        label=0
        for i in d:
            if num1<d[i].shape[0]:
                num1=d[i].shape[0]
                label=i
        return label



    #查找索引
    def findIndex(self,att,attname):
        # 属性所在的索引位置
        index = -1
        for j in range(len(att)):
            if att[j] == attname:
                index = j
        return index

    # 生成树
    #参数：去掉标签的属性集合，数据集合，工具类，父亲节点集合中哪个类样本最多的标签,父亲节点，父亲节点属性的取值
    def generateTree(self,attribute, data, tls, label, node1, LinkD):
        #更具标签划分当前集合；用来判断3种情况
        b, d = tls.getattributeD(np.array(data), self.label)
        if len(attribute) == 0 or tls.eampleissample(data):
            node3 = nde.node(self.label, tls.backMostLabel(data), LinkD)
            node3.set_isthref(True)
            node1.addchildrens(node3)
        elif len(b) == 1:
            node2 = nde.node(self.label, d[0], LinkD)
            node2.set_isthref(True)
            node1.addchildrens(node2)
        elif len(data) == 0:
            node4 = nde.node(self.label, label, LinkD)
            node4.set_isthref(True)
            node1.addchildrens(node4)
        else:
            MaxEntG = 0
            attName = 0
            t = 0

            #找到哪个属性信息增益最大
            for i in range(len(attribute)):
                EntG, t1 = tls.inforGain(attribute[i], data)
                if MaxEntG < EntG:
                    MaxEntG = EntG
                    attName = attribute[i]
                    t = t1
            node5 = nde.node(attName, t, LinkD)
            node5.set_isthref(False)
            node1.addchildrens(node5)

            #信息增益属性非连续属性；递归时属性集合中去掉改属性
            if t == -1:
                Datadict, datanum = tls.getattributeD(data, attName)
                attri = np.delete(np.array(attribute), tls.findIndex(np.array(attribute), attName))
                label = tls.backMostLabel(data)
                for i in Datadict:
                    data1 = np.array(Datadict[i])
                    self.generateTree(attri, data1, tls, label, node5, i)
            # 信息增益属性非连续属性；递归时属性集合中不去掉改属性
            else:
                Datadict1 = tls.getattributeDByNumber(data, attName, t)
                labe2 = tls.backMostLabel(data)
                for i in Datadict1:
                    data2 = np.array(Datadict1[i])
                    self.generateTree(attribute, data2, tls, labe2, node5, i)


    #打印树
    def printTree(self,node,ceng):
        print('层数%d,%s'%(ceng , node.get_attribute()))
        ceng+=1
        for i in node.get_childrens():

            self.printTree(i,ceng)

    # 预测树
    def predicate(self,node, data):

        label=0
        labeldata=0
        fname, fdata, ffatherdata = node.get_attribute()
        if node.get_isthref():
            label=fname
            labeldata=fdata
        else :
            node1 = (node.get_childrens())
            findex = self.findIndex(self.attribute, fname)
            for i in node1:
                cname, cdata, cfatherdata = i.get_attribute()
                if fdata == -1:

                    if cfatherdata == data[findex]:
                        label, labeldata=self.predicate(i, data)
                else:
                    if fdata > float(data[findex] )and cfatherdata == '-':
                        label, labeldata=self.predicate(i, data)
                    if fdata <= float(data[findex] ) and cfatherdata == '+':
                        label, labeldata=self.predicate(i, data)

        return label, labeldata