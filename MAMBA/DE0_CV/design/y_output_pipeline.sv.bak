module y_output_pipeline
    #(
    parameter integer C_SIZE          = 16,
    parameter integer D_SIZE          = 16,
    parameter integer U_SIZE          = 16,
    parameter integer STATE_SIZE      = 32,
    parameter integer Y_SIZE          = 32,
    parameter integer L               = 4,
    parameter integer D_IN            = 32,
    parameter integer N               = 16,
    parameter integer C_FRAC_BITS     = 8,
    parameter integer D_FRAC_BITS     = 8,
    parameter integer U_FRAC_BITS     = 8,
    parameter integer STATE_FRAC_BITS = 16,
    parameter integer Y_FRAC_BITS     = 16,
    parameter integer L_W             = (L    <= 1) ? 1 : $clog2(L),
    parameter integer D_W             = (D_IN <= 1) ? 1 : $clog2(D_IN),
    parameter integer N_W             = (N    <= 1) ? 1 : $clog2(N)
    )
    (
    input  logic clk,
    input  logic rst,

    // x_new(l,d,n) stream，順序固定為 l -> d -> n，n 最快。
    input  logic in_valid,
    input  logic [L_W-1:0] in_l,
    input  logic [D_W-1:0] in_d,
    input  logic [N_W-1:0] in_n,
    input  logic signed [STATE_SIZE-1:0] state_value,

    input  logic signed [C_SIZE-1:0] c_value,
    input  logic signed [D_SIZE-1:0] d_value,
    input  logic signed [U_SIZE-1:0] u_value,

    // 每累加完 N 個 state，輸出一筆 y(l,d)。
    output logic out_valid,
    output logic [L_W-1:0] out_l,
    output logic [D_W-1:0] out_d,
    output logic signed [Y_SIZE-1:0] out_data,
    output logic finish
    );

    localparam integer CX_MUL_W = C_SIZE + STATE_SIZE;
    localparam integer DU_MUL_W = D_SIZE + U_SIZE;

    localparam integer CX_TO_Y_SHIFT =
        C_FRAC_BITS + STATE_FRAC_BITS - Y_FRAC_BITS;
    localparam integer DU_TO_Y_SHIFT =
        D_FRAC_BITS + U_FRAC_BITS - Y_FRAC_BITS;

    localparam integer N_GROWTH = (N <= 1) ? 0 : $clog2(N);
    localparam integer ACC_W    = Y_SIZE + N_GROWTH + 1;

    // 目前版本的 fixed_round_shift 只支援右移或不移動。
    initial begin
        if (CX_TO_Y_SHIFT < 0)
            $error("y_output_pipeline: CX_TO_Y_SHIFT must be >= 0");
        if (DU_TO_Y_SHIFT < 0)
            $error("y_output_pipeline: DU_TO_Y_SHIFT must be >= 0");
    end

    // ---------------------------------------------------------------
    // Stage 0：C*x 與 D*u 乘法
    // D*u 對同一個 (l,d) 只需要在 n=0 時計算一次。
    // ---------------------------------------------------------------
    logic v0;
    logic [L_W-1:0] l0;
    logic [D_W-1:0] d0;
    logic [N_W-1:0] n0;
    logic signed [CX_MUL_W-1:0] cx_product0;
    logic signed [DU_MUL_W-1:0] du_product0;

    logic signed [Y_SIZE-1:0] cx_scaled0;
    logic signed [Y_SIZE-1:0] du_scaled0;

    fixed_round_shift #(
        .IN_WIDTH (CX_MUL_W),
        .OUT_WIDTH(Y_SIZE),
        .SHIFT    (CX_TO_Y_SHIFT)
    ) scale_cx (
        .in_data (cx_product0),
        .out_data(cx_scaled0)
    );

    fixed_round_shift #(
        .IN_WIDTH (DU_MUL_W),
        .OUT_WIDTH(Y_SIZE),
        .SHIFT    (DU_TO_Y_SHIFT)
    ) scale_du (
        .in_data (du_product0),
        .out_data(du_scaled0)
    );

    // ---------------------------------------------------------------
    // Stage 1：沿 n 累加 C(l,n)*x(l,d,n)，最後加 D(d)*u(l,d)
    // y(l,d) = sum_n C(l,n)*x_new(l,d,n) + D(d)*u(l,d)
    // ---------------------------------------------------------------
    logic signed [ACC_W-1:0] acc_reg;
    logic signed [Y_SIZE-1:0] du_cache;

    logic signed [ACC_W-1:0] cx_ext;
    logic signed [ACC_W-1:0] du_current_ext;
    logic signed [ACC_W-1:0] du_cache_ext;
    logic signed [ACC_W-1:0] acc_with_current;
    logic signed [ACC_W-1:0] y_full;

    always_comb begin
        cx_ext = {{(ACC_W-Y_SIZE){cx_scaled0[Y_SIZE-1]}}, cx_scaled0};
        du_current_ext =
            {{(ACC_W-Y_SIZE){du_scaled0[Y_SIZE-1]}}, du_scaled0};
        du_cache_ext =
            {{(ACC_W-Y_SIZE){du_cache[Y_SIZE-1]}}, du_cache};

        if (n0 == 0)
            acc_with_current = cx_ext;
        else
            acc_with_current = acc_reg + cx_ext;

        // N=1 時 n=0 同時也是最後一項，必須使用本筆的 D*u，
        // 不能使用前一組留下的 du_cache。
        if ((n0 == 0) && (n0 == N-1))
            y_full = acc_with_current + du_current_ext;
        else
            y_full = acc_with_current + du_cache_ext;
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            v0          <= 1'b0;
            l0          <= 0;
            d0          <= 0;
            n0          <= 0;
            cx_product0 <= 0;
            du_product0 <= 0;
            acc_reg     <= 0;
            du_cache    <= 0;
            out_valid   <= 1'b0;
            out_l       <= 0;
            out_d       <= 0;
            out_data    <= 0;
            finish      <= 1'b0;
        end
        else begin
            out_valid <= 1'b0;
            finish    <= 1'b0;

            // Stage 0
            v0 <= in_valid;
            if (in_valid) begin
                l0 <= in_l;
                d0 <= in_d;
                n0 <= in_n;

                cx_product0 <= $signed(c_value) * $signed(state_value);

                if (in_n == 0) begin
                    du_product0 <= $signed(d_value) * $signed(u_value);
                end
            end

            // Stage 1
            if (v0) begin
                if (n0 == 0)
                    du_cache <= du_scaled0;

                if (n0 == N-1) begin
                    // 與現有 state_update_stage 相同，超出 Y_SIZE 時採截位。
                    out_valid <= 1'b1;
                    out_l     <= l0;
                    out_d     <= d0;
                    out_data  <= y_full[Y_SIZE-1:0];
                    acc_reg   <= '0;

                    if ((l0 == L-1) && (d0 == D_IN-1))
                        finish <= 1'b1;
                end
                else begin
                    acc_reg <= acc_with_current;
                end
            end
        end
    end

endmodule
