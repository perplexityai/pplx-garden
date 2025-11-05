pub struct Defer<F: FnMut()> {
    f: F,
    canceled: bool,
}

impl<F: FnMut()> Defer<F> {
    pub fn new(f: F) -> Self {
        Self { f, canceled: false }
    }

    pub fn cancel(&mut self) {
        self.canceled = true;
    }
}

impl<F: FnMut()> Drop for Defer<F> {
    fn drop(&mut self) {
        if !self.canceled {
            (self.f)();
        }
    }
}
