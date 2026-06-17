`timescale 1ns/1ps

module testbench();

	//=======================================================
	//  Parameters & Declarations
	//=======================================================
	
	// DUT Parameters (與你的模組對齊)
	parameter A_size     = 16;
	parameter B_size     = 16;
	parameter Delta_size = 16;
	parameter L          = 4;
	parameter D_IN       = 32;
	parameter N          = 16;
	
	// 系統控制參數
	parameter NUM_TESTS  = 1; // 測試檔案的數量

	// 系統訊號
	logic clk;
	logic rst;
	
	// DUT 輸入訊號
	logic start;
	logic [0:15] data;
	
	// DUT 輸出訊號
	logic finish;
	
	// 預留給你「等等寫」的輸出訊號（用來寫入檔案）
	logic out_valid;         // 標示 output data 是否有效
	logic [15:0] out_data;   // 運算結果輸出

	// 測試環境變數
	integer out_result_file;
	integer test_idx;
	bit test_result;
	bit all_pass;

	//=======================================================
	//  DUT Instantiation
	//=======================================================
	Discretization #(
		.A_size(A_size),
		.B_size(B_size),
		.Delta_size(Delta_size),
		.L(L),
		.D_IN(D_IN),
		.N(N)
	) dut (	
		.clk(clk),
		.rst(rst),
		.start(start),
		.data(data),
		
		.finish(finish)
		// 這裡未來記得接上你的輸出訊號
		// .out_valid(out_valid),
		// .out_data(out_data)
	);

	//=======================================================
	//  Tasks
	//=======================================================

	// 1. 讀取檔案並將資料送入 DUT
	task automatic feed_data_from_file(input string filename);
		integer file, status;
		string line;
		int line_num;
		logic [15:0] read_data;

		file = $fopen(filename, "r");
		if (file == 0) begin
			$display("ERROR: Failed to open %s", filename);
			$finish;
		end

		// 啟動 DUT (給一個 cycle 的 start pulse)
		@(negedge clk);
		start = 1;
		@(negedge clk);
		start = 0;

		line_num = 0;
		// 根據你的 FSM，進入狀態 A、B、DELTA 時需要連續餵資料
		while (!$feof(file)) begin
			status = $fgets(line, file);
			if (status == 0) continue; // 跳過空行

			// 解析 16 進位的字串 (測資檔案每行應為一個 16-bit hex)
			status = $sscanf(line, "%h", read_data);
			if (status == 1) begin
				data = read_data;
				@(negedge clk); // 等待下一個負緣再換下一筆資料
			end else if (line != "\n" && line != "\r\n" && line != "") begin
				$display("WARNING: Failed to parse line %0d: \"%s\"", line_num, line);
			end
			line_num++;
		end

		$fclose(file);
		data = 0; // 餵完資料後歸零
	endtask

	// 2. 比對輸出的結果檔與標準答案檔
	task automatic compare_files(input string file1, input string file2, output bit is_equal);
		int f1, f2;
		string line1, line2;
		is_equal = 1;

		f1 = $fopen(file1, "r");
		f2 = $fopen(file2, "r");

		if (f1 == 0 || f2 == 0) begin
			$display("ERROR: Cannot open file(s) for comparison.");
			is_equal = 0;
			return;
		end

		while (!$feof(f1) || !$feof(f2)) begin
			$fgets(line1, f1);
			$fgets(line2, f2);

			if (line1 != line2) begin
				$display("Mismatch found:\n  Output: %s  Answer: %s", line1, line2);
				is_equal = 0;
				break;
			end
		end

		if (!$feof(f1) || !$feof(f2)) begin
			$display("Files have different lengths.");
			is_equal = 0;
		end

		$fclose(f1);
		$fclose(f2);
	endtask

	// 3. 封裝單次測試流程
	task automatic run_one_test(
		input int test_id,
		input string input_file,
		input string output_file,
		input string answer_file,
		output bit single_pass
	);
		$display("========== Test %0d start ==========", test_id);
		$display("Input : %s", input_file);
		$display("Output: %s", output_file);
		$display("Answer: %s", answer_file);

		// 開啟準備寫入的結果檔
		out_result_file = $fopen(output_file, "w");
		if (!out_result_file) begin
			$display("ERROR: Cannot open %s for writing.", output_file);
			$finish;
		end

		// 餵資料
		feed_data_from_file(input_file);

		// 等待 FSM 跑到 FINISH 狀態
		wait(finish == 1'b1);
		@(negedge clk); 

		// 關閉結果檔
		$fclose(out_result_file);
		out_result_file = 0;

		// 進行比對
		compare_files(output_file, answer_file, single_pass);
		if (single_pass)
			$display("Test %0d PASS.\n", test_id);
		else
			$display("Test %0d FAIL.\n", test_id);
	endtask

	//=======================================================
	//  Clock & Output Capture
	//=======================================================
	
	// Clock Generation (50MHz)
	always #10 clk = ~clk;

	// 當 out_valid 為 High 時，把資料寫入 txt
	// (這個 block 完美複製了你範例中寫入 conv_result 的邏輯)
	always @(negedge clk) begin
		if (out_valid && out_result_file != 0) begin
			$fwrite(out_result_file, "%04h\n", out_data);
		end
	end
	
	//=======================================================
	//  Main Process
	//=======================================================
	initial begin
		// 初始化訊號
		clk = 0;
		rst = 0;
		start = 0;
		data = 0;
		out_valid = 0; // 等你的 DUT 寫好後，這行就可以刪除，由 DUT 驅動
		out_data = 0;  // 等你的 DUT 寫好後，這行就可以刪除，由 DUT 驅動
		
		out_result_file = 0;
		all_pass = 1;
		
		// Reset Sequence
		rst = 1;
		#100;
		rst = 0;
		#100;
		
		// 自動化測試迴圈
		for (test_idx = 0; test_idx < NUM_TESTS; test_idx = test_idx + 1) begin
			run_one_test(
				test_idx,
				$sformatf("../tb/test_in_%0d.txt", test_idx),       // 輸入的測資
				$sformatf("../tb/test_out_%0d.txt", test_idx),      // 產生的結果
				$sformatf("../tb/test_answer_%0d.txt", test_idx),   // 標準答案
				test_result
			);
			all_pass = all_pass && test_result;
		end

		// 總結測試結果
		if (all_pass)
			$display("\n[SUCCESS] All %0d tests passed!", NUM_TESTS);
		else
			$display("\n[FAILED] One or more tests failed.");
		
		$stop;
	end

endmodule