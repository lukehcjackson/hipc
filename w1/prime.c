int prime(int n) {

    int sum = 0;
    int flag = 0;

    for (int i = 2; i <= n; i++) {

        flag = 1;

        for (int j = 2; j < n; j++) {
            if (n % j == 0) {
                flag = 0;
            }
        }  

        if (flag) {
            sum += i;
        }
    }

    return sum;
}