module state_update_stage 
	#(
    parameter integer STATE_SIZE = 32,
    parameter integer L          = 4,
    parameter integer D_IN       = 32,
    parameter integer N          = 16,
    parameter integer L_W        = (L    <= 1) ? 1 : $clog2(L),
    parameter integer D_W        = (D_IN <= 1) ? 1 : $clog2(D_IN),
    parameter integer N_W        = (N    <= 1) ? 1 : $clog2(N)
	) 
	(
    input  logic clk,
    input  logic rst,

    input  logic a_valid,
    input  logic [L_W-1:0] a_l,
    input  logic [D_W-1:0] a_d,
    input  logic [N_W-1:0] a_n,
    input  logic signed [STATE_SIZE-1:0] a_data,

    input  logic b_valid,
    input  logic [L_W-1:0] b_l,
    input  logic [D_W-1:0] b_d,
    input  logic [N_W-1:0] b_n,
    input  logic signed [STATE_SIZE-1:0] b_data,

    output logic write_valid,
    output logic [D_W-1:0] write_d,
    output logic [N_W-1:0] write_n,
    output logic signed [STATE_SIZE-1:0] write_data,

    output logic out_valid,
    output logic signed [STATE_SIZE-1:0] out_data,
    output logic finish,
    output logic alignment_error
	);

    logic tags_match;
    logic signed [STATE_SIZE:0] sum_full;

    always_comb begin
        tags_match = (a_l == b_l) && (a_d == b_d) && (a_n == b_n);
        sum_full   = $signed(a_data) + $signed(b_data);
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            write_valid     <= 1'b0;
            write_d         <= '0;
            write_n         <= '0;
            write_data      <= '0;
            out_valid       <= 1'b0;
            out_data        <= '0;
            finish          <= 1'b0;
            alignment_error <= 1'b0;
        end
        else begin
            write_valid <= 1'b0;
            out_valid   <= 1'b0;
            finish      <= 1'b0;

            if ((a_valid != b_valid) || (a_valid && b_valid && !tags_match)) begin
                alignment_error <= 1'b1;
            end

            if (a_valid && b_valid && tags_match) begin
                write_valid <= 1'b1;
                write_d     <= a_d;
                write_n     <= a_n;
                write_data  <= sum_full[STATE_SIZE-1:0];

                out_valid <= 1'b1;
                out_data  <= sum_full[STATE_SIZE-1:0];

                if ((a_l == L-1) && (a_d == D_IN-1) && (a_n == N-1)) begin
                    finish <= 1'b1;
                end
            end
        end
    end

endmodule
