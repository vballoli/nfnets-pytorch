from nfnets import WSConv2d

def test_wsconv2d():
    c = WSConv2d(3,6,3)
    c(torch.randn(1,3,32,32))