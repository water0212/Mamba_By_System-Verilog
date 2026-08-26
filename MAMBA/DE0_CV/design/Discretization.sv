module Discretization #(
    parameter integer A_size     = 16,
    parameter integer B_size     = 16,
    parameter integer Delta_size = 16,
    parameter integer U_size     = 16,
	 parameter integer C_size     = 16,
	 parameter integer D_size     = 16,
    parameter integer L          = 32,
    parameter integer D_IN       = 32,
    parameter integer N          = 16,

    // 預設假設 A、B、delta、u 都是 Q8，state_x 是 Q16。
    parameter integer A_FRAC_BITS     = 0,
    parameter integer B_FRAC_BITS     = 8,
    parameter integer DELTA_FRAC_BITS = 8,
    parameter integer U_FRAC_BITS     = 8,
    parameter integer STATE_FRAC_BITS = 16,
    parameter integer EXP_FRAC_BITS   = 8
) (
    input  logic clk,
    input  logic rst,
    input  logic start,
    input  logic [15:0] data,

    output logic out_valid,
    output logic [31:0] x_out_data,
	 output logic [31:0] y_out_data,

    // 保留舊介面：所有輸入載入完成、開始 scan pipeline 時 pulse 一拍。
    output logic start_delta_mul,
    output logic finish
);

    logic load_busy;
    logic load_done;
    logic pipeline_busy;
    logic pipeline_out_valid;
    logic signed [31:0] x_pipeline_out_data;
	 logic signed [31:0] y_pipeline_out_data;
    logic pipeline_finish;
    logic alignment_error;

    logic signed [A_size-1:0]     reg_A     [0:D_IN-1][0:N-1];
    logic signed [B_size-1:0]     reg_B     [0:L-1][0:N-1];
    logic signed [Delta_size-1:0] reg_delta [0:L-1][0:D_IN-1];
    logic signed [U_size-1:0]     reg_u     [0:L-1][0:D_IN-1];
	 logic signed [C_size-1:0]     reg_c     [0:L-1][0:N-1];
    logic signed [D_size-1:0]     reg_d     [0:D_IN-1];
	 
    logic accepted_start;
    assign accepted_start = start && !load_busy && !pipeline_busy;

    input_loader #(
        .A_SIZE    (A_size),
        .B_SIZE    (B_size),
        .DELTA_SIZE(Delta_size),
        .U_SIZE    (U_size),
        .L         (L),
        .D_IN      (D_IN),
        .N         (N)
    ) loader (
        .clk      (clk),
        .rst      (rst),
        .start    (accepted_start),
        .data     (data),
        .busy     (load_busy),
        .load_done(load_done),
        .reg_A    (reg_A),
        .reg_B    (reg_B),
        .reg_delta(reg_delta),
        .reg_u    (reg_u),
		  .reg_c		(reg_c),
		  .reg_d		(reg_d)
    );

    mamba_scan_pipeline #(
        .A_SIZE         (A_size),
        .B_SIZE         (B_size),
        .DELTA_SIZE     (Delta_size),
        .U_SIZE         (U_size),
        .STATE_SIZE     (32),
        .L              (L),
        .D_IN           (D_IN),
        .N              (N),
        .A_FRAC_BITS    (A_FRAC_BITS),
        .B_FRAC_BITS    (B_FRAC_BITS),
        .DELTA_FRAC_BITS(DELTA_FRAC_BITS),
        .U_FRAC_BITS    (U_FRAC_BITS),
        .STATE_FRAC_BITS(STATE_FRAC_BITS),
        .EXP_FRAC_BITS  (EXP_FRAC_BITS)
    ) scan_pipeline (
        .clk            (clk),
        .rst            (rst),
        .start          (load_done),
        .reg_A          (reg_A),
        .reg_B          (reg_B),
        .reg_delta      (reg_delta),
        .reg_u          (reg_u),
		  .reg_c          (reg_c),
		  .reg_d          (reg_d),
        .busy           (pipeline_busy),
        .out_valid      (pipeline_out_valid),
        .x_out_data     (x_pipeline_out_data),
		  .y_out_data     (y_pipeline_out_data),
        .finish         (pipeline_finish),
        .alignment_error(alignment_error)
    );

    assign start_delta_mul = load_done;
    assign out_valid       = pipeline_out_valid;
    assign x_out_data      = x_pipeline_out_data;
	 assign y_out_data      = y_pipeline_out_data;
    assign finish          = pipeline_finish;

endmodule
