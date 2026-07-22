module exp_state_mul_stage 
    #(
    parameter integer EXP_SIZE      = 32,
    parameter integer STATE_SIZE    = 32,
    parameter integer EXP_FRAC_BITS = 8,
    parameter integer L_W           = 2,
    parameter integer D_W           = 5,
    parameter integer N_W           = 4
    ) 
    (
    input  logic clk,
    input  logic rst,

    input  logic in_valid,
    input  logic [L_W-1:0] in_l,
    input  logic [D_W-1:0] in_d,
    input  logic [N_W-1:0] in_n,
    input  logic [EXP_SIZE-1:0] exp_value,
    input  logic signed [STATE_SIZE-1:0] state_value,

    output logic out_valid,
    output logic [L_W-1:0] out_l,
    output logic [D_W-1:0] out_d,
    output logic [N_W-1:0] out_n,
    output logic signed [STATE_SIZE-1:0] out_data
    );

    localparam integer MUL_W = EXP_SIZE + 1 + STATE_SIZE;

    logic signed [EXP_SIZE:0] exp_signed;
    logic signed [MUL_W-1:0] mul_full;
    logic signed [STATE_SIZE-1:0] mul_scaled;

    assign exp_signed = $signed({1'b0, exp_value});
    assign mul_full   = exp_signed * $signed(state_value);

    fixed_round_shift #(
        .IN_WIDTH  (MUL_W),
        .OUT_WIDTH (STATE_SIZE),
        .SHIFT     (EXP_FRAC_BITS)
    ) scale_exp_x (
        .in_data  (mul_full),
        .out_data (mul_scaled)
    );

    always_ff @(posedge clk) begin
        if (rst) begin
            out_valid <= 1'b0;
            out_l     <= '0;
            out_d     <= '0;
            out_n     <= '0;
            out_data  <= '0;
        end
        else begin
            out_valid <= in_valid;
            if (in_valid) begin
                out_l    <= in_l;
                out_d    <= in_d;
                out_n    <= in_n;
                out_data <= mul_scaled;
            end
        end
    end

endmodule
