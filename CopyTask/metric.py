import torch

def accuracy(predicted_item,target_item):
    score = torch.argmax(predicted_item, dim=2)[:,-10:]
    target = target_item[:,-10:]

    batch_size = target.size(0)
    accu1 = torch.true_divide(torch.sum(score==target),(batch_size*10))
    count2 = 0
    for i in range(batch_size): #batch size
        if all(score[i]==target[i]):
            count2+=1
    accu2 = torch.true_divide(count2,batch_size)
    return accu1, accu2


def accuracy_varyT(predicted_item,target_item, T_length):
    accu1_sum=0
    count2=0 
    batch_size = target_item.size(0)

    for i in range(batch_size):
        score = torch.argmax(predicted_item, dim=2)[i, T_length[i]+9:T_length[i]+19]
        target = target_item[i, T_length[i]+9:T_length[i]+19]
        accu1_sum+=torch.sum(score==target)
        if all(score==target):
            count2+=1 

    accu1 = accu1_sum/(batch_size*10)
    accu2 = count2/batch_size

    return accu1, accu2
