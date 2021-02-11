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

