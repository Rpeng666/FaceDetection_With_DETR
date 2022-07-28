import torch
from build_detr import DETR
from loss import Loss_fn
from torch.optim import Adam, SGD, AdamW
from datasets import get_dataloader



def main():
    num_classes = 1
    lr = 0.001
    epochs = 100
    batch_size = 30
    weight_dict = {'loss_ce': 1., 'loss_bbox': 5., 'loss_giou': 2.}

    log_file = open('./main.out', 'w', encoding='utf-8')

    device = torch.device("cuda")

    loss_fn = Loss_fn(cost_weight_dict = weight_dict,device=device)

    trainloader, testloader = get_dataloader(batch_size= batch_size)
    print("数据载入成功")

    model = DETR(num_classes, 10, device).to(device)
    model.load_state_dict(torch.load('./model.pth'))

    print('模型生成成功')

    optimizer = AdamW(model.parameters(), lr = 0.001, weight_decay= 0.0001)
    # optimizer = SGD(model.parameters(), lr = lr)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100)

    for epoch in range(epochs):
        model.train()
        # train
        for i, (data, target) in enumerate(trainloader):

            data = data.to(device)
            # data:[batch_size, 3, img_h, img_w]
            # target: list of bbox tensors
            outputs = model(data) # {'pred_logits':[batch, num_queries, 2], 'pred_boxes': [batch, num_queries, 4]}
            
            loss_ce ,  loss_bbox , loss_giou = loss_fn(outputs['pred_logits'], outputs['pred_boxes'], target)

            loss = ( loss_ce * weight_dict['loss_ce'] + loss_bbox * weight_dict['loss_bbox'] + loss_giou * weight_dict['loss_giou'] ).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if( i % 5 == 0):
                print(f'epoch:{epoch} batch:{i}  loss_ce:{loss_ce}   loss_bbox:{loss_bbox}    loss_giou:{loss_giou}    train_loss:{loss}')
                print(f'epoch:{epoch} batch:{i}  loss_ce:{loss_ce}   loss_bbox:{loss_bbox}    loss_giou:{loss_giou}    train_loss:{loss}', file= log_file, flush=True)
                # print(outputs['pred_logits'])
        
        lr_scheduler.step()
          
        # eval
        # model.eval()

        # with torch.no_grad():
        #     loss_list = []
        #     for i, (data, target) in enumerate(testloader):
        #         data = data.to(device)

        #         outputs = model(data)

        #         loss_ce ,  loss_bbox , loss_giou = loss_fn(outputs['pred_logits'], outputs['pred_boxes'], target)

        #         loss = ( loss_ce * weight_dict['loss_ce'] + loss_bbox * weight_dict['loss_bbox'] + loss_giou * weight_dict['loss_giou'] ).mean()
        #         loss_list.append(loss)

        #     loss = sum(loss_list)/len(loss_list)

        #     print(f'epoch:{epoch} test_loss:{loss}')
        #     print(f'epoch:{epoch} test_loss:{loss}', file= log_file, flush= True)

    torch.save(model.state_dict(), 'model.pth')
    log_file.close()

if __name__ == '__main__':
    main()