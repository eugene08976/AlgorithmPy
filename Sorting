class NTHUList(list):
    
    def selectSort(self):
        '''Select Sort'''
        
        for i in range(len(self)-1):
            for j in range(i+1, len(self)):
                if self[i] < self[j]:
                    self[i], self[j] = self[j], self[i]
        return
    
    def bubbleSort(self):
        '''Bubble Sort'''
        
        if len(self) <= 1:
            return
        for i in range(1, len(self)):
            j = i
            while j > 0 and self[j] > self[j-1]:
                self[j], self[j-1] = self[j-1], self[j]
                j = j-1
        return
    
    
    def mergeSort(self, inputList):
        '''Merge Sort, Take List as argument'''
        
        if len(inputList) <= 1:
            return inputList
        left = inputList[:len(inputList)//2]
        right = inputList[len(inputList)//2:]
        
        sortedLeft = self.mergeSort(left)
        sortedRight = self.mergeSort(right)
        
        returnList = []
        ri = 0
        li = 0
        while ri < len(sortedRight) and li < len(sortedLeft):
            if sortedLeft[li] > sortedRight[ri]:
                returnList.append(sortedLeft[li])
                li = li+1
            else:
                returnList.append(sortedRight[ri])
                ri = ri+1
        if ri == len(sortedRight):
            returnList.extend(sortedLeft[li:])
        else:
            returnList.extend(sortedRight[ri:])
            
        return returnList
                
        
    def mSort(self):
        '''Merge Sort'''
        
        if len(self) <= 1:
            return self
        left = NTHUList(self[:len(self)//2])
        right = NTHUList(self[len(self)//2:])
        
        sortedLeft = left.mSort()
        sortedRight = right.mSort()
        
        returnList = NTHUList()
        ri, li = 0, 0
        while ri < len(sortedRight) and li < len(sortedLeft):
            if sortedLeft[li] > sortedRight[ri]:
                returnList.append(sortedLeft[li])
                li = li+1
            else:
                returnList.append(sortedRight[ri])
                ri = ri+1
                
        if ri == len(sortedRight):
            returnList.extend(sortedLeft[li:])
        else:
            returnList.extend(sortedRight[ri:])
            
        return returnList
    
    def heapSort(self):
        '''Heap Sort, Recursive Version'''
    
        def heapify(self):
            for i in range(len(self)):
                shiftup(self, i)
            return
    
        def shiftup(self, i):
            if i == 0:
                return
            else:
                if self[i] <= self[(i-1)//2]:
                    return
                else:
                    self[i], self[(i-1)//2] = self[(i-1)//2], self[i]
                    shiftup(self, (i-1)//2)
                
        def shiftdown(self, i, end):
            lc = 2*i + 1
            rc = 2*i + 2
            if lc > end:
                return
            elif lc == end:
                if self[i] < self[lc]:
                    self[i], self[lc] = self[lc], self[i]
                return
            else:
                if self[i] < max(self[lc], self[rc]):
                    if self[lc] > self[rc]:
                        self[i], self[lc] = self[lc], self[i]
                        shiftdown(self, lc, end)
                    else:
                        self[rc], self[i] = self[i], self[rc]
                        shiftdown(self, rc, end)
                return
        
        # Main routine for heapSort
        heapify(self)
        end = len(self)-1
        while end > 0:
            self[0], self[end] = self[end], self[0]
            end -= 1
            shiftdown(self, 0, end)
        return  
    
    def heapSortNonRecur(x):
        '''HeapSort Non-Recursive Version'''
    
        length = len(x)
        if length <= 1:
            return
    
        '''Heapify'''
        for i in range(length):
            j = i
            while j > 0 and x[j] < x[(j-1)//2]:
                x[j], x[(j-1)//2] = x[(j-1)//2], x[j]
                j = (j-1)//2
            
        for i in range(length-1, 0, -1):
            x[i], x[0] = x[0], x[i]
        
            '''Shift Down'''
            j = 0
            last = i-1
            while j * 2 + 1 <= last: # as long as j has child
                lc = j * 2 + 1
                rc = j * 2 + 2
                if lc == last: # There is only one child (left child)
                    if x[j] > x[lc]:
                        x[j], x[lc] = x[lc], x[j]
                        j = lc
                    else:
                        break
                else: # Consider both children
                    if x[j] > min(x[lc], x[rc]):
                        if x[lc] < x[rc]:
                            x[j], x[lc] = x[lc], x[j]
                            j = lc
                        else:
                            x[j], x[rc] = x[rc], x[j]
                            j = rc
                    else:
                        break
        return
                
    def quickSort(self, first, last):
        '''Quick Sort, an in-place sorter'''
        
        if first >= last:
            return
        
        # Pick the middle one to be the pivot
        middle = (first+last)//2
        self[first], self[middle] = self[middle], self[first]
        pivot = self[first]
        
        # March towards the center
        left, right = first + 1, last
        while left < right:
            while left < right and self[left] < pivot:
                left += 1
            while left < right and self[right] >= pivot:
                right -= 1
            self[left], self[right] = self[right], self[left] 
            
        center = left if self[left] < pivot else left - 1
        self[first], self[center] = self[center], self[first]
        self.quickSort(first, center - 1)
        self.quickSort(center + 1, last)
        
        return  
        
        
