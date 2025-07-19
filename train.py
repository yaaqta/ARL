import HEMS
manager = HEMS.HEMS()
manager.train()
manager.save('saved_nets/my_net')