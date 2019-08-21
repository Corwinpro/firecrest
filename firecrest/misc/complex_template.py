class Complex:
    def __init__(self, real, imag=None):
        self.real = real
        if imag is None:
            self.imag = self.real
        else:
            self.imag = imag
