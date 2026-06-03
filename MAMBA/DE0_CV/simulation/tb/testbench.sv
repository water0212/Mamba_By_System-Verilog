`timescale 1ns/1ps

module testbench;

    parameter A_size     = 16;
    parameter B_size     = 16;
    parameter delta_size = 16;
    parameter L          = 4;
    parameter D_IN       = 32;
    parameter N          = 16;

    logic clk;
    logic rst;
    logic start;
    logic finish;
    logic [0:15] data;

    // DUT
    Discretization #(
        .A_size(A_size),
        .B_size(B_size),
        .delta_size(delta_size),
        .L(L),
        .D_IN(D_IN),
        .N(N)
    ) dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .data(data),
        // output 你還沒寫，所以這裡先不接
        .finish(finish)
    );

    // clock：10ns period
    always #5 clk = ~clk;

    integer i;

    initial begin
        clk   = 0;
        rst   = 1;
        start = 0;
        data  = 0;

        // reset
        #20;
        rst = 0;

        // start pulse
        @(posedge clk);
        start = 1;

        @(posedge clk);
        start = 0;

        // =========================
        // input A：512筆
        // =========================
        for (i = 0; i < D_IN * N; i = i + 1) begin
            @(posedge clk);
            data = i;
        end

        // =========================
        // input B：64筆
        // =========================
        for (i = 0; i < L * N; i = i + 1) begin
            @(posedge clk);
            data = 16'h1000 + i;
        end

        // =========================
        // input delta：128筆
        // =========================
        for (i = 0; i < L * D_IN; i = i + 1) begin
            @(posedge clk);
            data = 16'h2000 + i;
        end

        // 等 FSM 跑完
        repeat (20) @(posedge clk);

        // 檢查部分資料
        $display("===== Check reg_A =====");
        $display("reg_A[0][0]  = %h", dut.reg_A[0][0]);
        $display("reg_A[1][0]  = %h", dut.reg_A[1][0]);
        $display("reg_A[0][1]  = %h", dut.reg_A[0][1]);
        $display("reg_A[31][15]= %h", dut.reg_A[31][15]);

        $display("===== Check reg_B =====");
        $display("reg_B[0][0]  = %h", dut.reg_B[0][0]);
        $display("reg_B[1][0]  = %h", dut.reg_B[1][0]);
        $display("reg_B[0][1]  = %h", dut.reg_B[0][1]);
        $display("reg_B[3][15] = %h", dut.reg_B[3][15]);

        $display("===== Check reg_delta =====");
        $display("reg_delta[0][0]  = %h", dut.reg_delta[0][0]);
        $display("reg_delta[1][0]  = %h", dut.reg_delta[1][0]);
        $display("reg_delta[0][1]  = %h", dut.reg_delta[0][1]);
        $display("reg_delta[3][31] = %h", dut.reg_delta[3][31]);

        $stop;
    end

endmodule