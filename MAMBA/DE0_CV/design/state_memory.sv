module state_memory 
	#(
    parameter integer D_IN       = 32,
    parameter integer N          = 16,
    parameter integer STATE_SIZE = 32,
    parameter integer D_W        = (D_IN <= 1) ? 1 : $clog2(D_IN),
    parameter integer N_W        = (N <= 1) ? 1 : $clog2(N)
    ) 
    (
    input  logic clk,
    input  logic rst,
    input  logic clear,

    input  logic [D_W-1:0] read_d,
    input  logic [N_W-1:0] read_n,
    

    input  logic write_valid,
    input  logic [D_W-1:0] write_d,
    input  logic [N_W-1:0] write_n,
    input  logic signed [STATE_SIZE-1:0] write_data,

    output logic signed [STATE_SIZE-1:0] read_data
    );

    logic signed [STATE_SIZE-1:0] x_mem [0:D_IN-1][0:N-1];
    integer d;
    integer n;

    always_comb begin
        read_data = x_mem[read_d][read_n];
    end

    always_ff @(posedge clk) begin
        if (rst || clear) begin
            for (d = 0; d < D_IN; d = d + 1) begin
                for (n = 0; n < N; n = n + 1) begin
                    x_mem[d][n] <= '0;
                end
            end
        end
        else if (write_valid) begin
            x_mem[write_d][write_n] <= write_data;
        end
    end

endmodule
