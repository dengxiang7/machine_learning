class node:

    def __init__(self, attributename, attributedata,fatherLinkData=None):
        self.attributename = attributename
        self._attributedata = attributedata
        self.childrens = []
        self.fatherLinkData=fatherLinkData
        self.isthref=False

    def addchildrens(self, node):
        self.childrens.append(node)

    def get_attribute(self):
        return self.attributename, self._attributedata,self.fatherLinkData

    def get_childrens(self):
        return self.childrens
    def get_isthref(self):
        return self.isthref
    def set_isthref(self,bl):
        self.isthref=bl

    def get_isthref(self):
        return self.isthref